"""UArmDevice: robosuite-compatible teleoperation device using the U-ARM servo arm.

Reads 8-servo data (7 arm joints + 1 gripper) from the physical U-ARM arm via
serial port and converts joint angle deltas to JOINT_POSITION actions for a
7-DOF Panda robot.

Control mode — JOINT_POSITION (robosuite):
    The JOINT_POSITION controller treats the action as a *position delta*
    relative to the current joint configuration, scaled by output_max=0.05 rad.
    To achieve 1:1 physical-to-simulation tracking:
        desired_joint_delta = delta_rad * sensitivity
        sim_position_change  = desired_joint_delta * output_max
    Setting sensitivity = 1/output_max = 1/0.05 = 20 gives 1° physical → 1° sim.
    Increase sensitivity to make the simulation arm move faster relative to the
    physical arm (>1:1 amplification); decrease for slower, finer control.

Servo layout (7-DOF + gripper, 8 servos total):
    servo 0-6  →  Panda joint 0-6 (direct 1:1, no reordering needed)
    servo 7    →  gripper (absolute angle, not delta)

Serial protocol:
    Hardware: Zhonglin servo, ASCII command format
    Baud rate: 115200
    PWM range: 500–2500 → 0–270 degrees

Controls:
    Move U-ARM joints  →  Panda joints follow
    Rotate servo 7     →  Simulation gripper opens / closes
    Press 'q'          →  End current episode

Requirements:
    pyserial  (pip install pyserial)
    pynput    (already installed by robosuite)

References:
    LeRobot-Anything-U-Arm/src/uarm/scripts/Follower_Arm/LeRobot/uarm.py
    LeRobot-Anything-U-Arm/src/simulation/teleop_sim.py
"""

import re
import time

import numpy as np
import serial
from pynput.keyboard import Listener as KeyboardListener
from robosuite.devices import Device


class UArmDevice(Device):
    """Teleoperation device backed by a U-ARM physical arm (Zhonglin servo protocol).

    Implements the robosuite ``Device`` interface so it can be dropped into
    ``collect_human_trajectory()`` alongside the existing Keyboard / SpaceMouse
    devices.  Because U-ARM natively produces joint-space data, the device is
    designed to work with the ``JOINT_POSITION`` controller.

    Usage in collect_demonstration.py::

        device = UArmDevice(port="/dev/ttyUSB0")
        device.start_control()          # called once per episode
        action, grasp = device.get_action(robot=active_robot)
    """

    # ── Servo protocol constants ──────────────────────────────────────────────
    _PWM_MIN: int = 500
    _PWM_MAX: int = 2500
    _ANGLE_RANGE_DEG: float = 270.0
    _SERIAL_DELAY: float = 0.008          # seconds between write and read_all

    # ── Gripper mapping constant ─────────────────────────────────────────────
    # Servo 6 angle range that spans full open ↔ full close.
    _GRIPPER_SERVO_RANGE_RAD: float = 1.5 * np.pi  # 270° expressed in radians

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        sensitivity: float = 20.0,
    ):
        """Initialise the device, open the serial port and calibrate zero angles.

        Args:
            port:        Serial device path (e.g. ``"/dev/ttyUSB0"``).
            baudrate:    Serial baud rate (default 115200 for Zhonglin servos).
            sensitivity: Per-radian scaling applied to joint angle deltas before
                         they are sent as JOINT_POSITION commands.
                         Default 20.0 = 1/output_max (output_max=0.05 in
                         robosuite's joint_position.json), which gives a 1:1
                         physical-to-simulation mapping for slow movements.
                         Increase for a larger / amplified simulation motion;
                         decrease for finer, smaller motion.
        """
        self.sensitivity = sensitivity

        # ── Open serial port ────────────────────────────────────────────────
        self._ser = serial.Serial(port, baudrate, timeout=0.1)
        print(f"[UArmDevice] Serial port opened: {port}")

        # ── Calibrate zero angles (one-time, at startup) ─────────────────
        self._zero_angles: list = [0.0] * 8
        self._init_servos()

        # ── Episode state ─────────────────────────────────────────────────
        # _prev_offsets is set in start_control() before each episode.
        self._prev_offsets: list = [0.0] * 8
        self._reset: bool = False

        # ── Keyboard listener: 'q' ends the current episode ──────────────
        self._kb_listener = KeyboardListener(on_press=self._on_key_press)
        self._kb_listener.daemon = True
        self._kb_listener.start()

    # ──────────────────────────────────────────────────────────────────────────
    #  Device ABC interface
    # ──────────────────────────────────────────────────────────────────────────

    def start_control(self) -> None:
        """Prepare the device for a new episode.

        Reads the current arm position and stores it as the baseline so that
        the first action delta is zero (no sudden jump at episode start).
        Call this once before each episode loop.
        """
        self._reset = False
        self._prev_offsets = self._read_offsets()
        print(
            "[UArmDevice] Control started.  Move the U-ARM to teleoperate.  "
            "Press 'q' to finish the episode."
        )

    def get_controller_state(self) -> dict:
        """Not used by UArmDevice.

        Raises:
            NotImplementedError: Always.  Use ``get_action()`` instead.
        """
        raise NotImplementedError(
            "UArmDevice does not implement get_controller_state(). "
            "Call get_action(robot=active_robot) directly."
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Main action interface
    # ──────────────────────────────────────────────────────────────────────────

    def get_action(self, robot=None):
        """Read the current U-ARM state and return a JOINT_POSITION action.

        Args:
            robot: The active robosuite ``Robot`` object.  Used to determine
                   gripper DOF so the action vector is correctly sized.
                   Defaults to 2 (PandaGripper) when ``None``.

        Returns:
            ``(action, grasp)`` where:
                - ``action`` is ``np.ndarray`` of shape ``(7 + gripper_dof,)``:
                    7 joint position-delta commands (each in ``[-1, 1]``) followed
                    by ``gripper_dof`` identical gripper position commands.
                    The JOINT_POSITION controller scales each element by
                    output_max (0.05 rad) before applying it as a position delta.
                - ``grasp`` is ``1`` (close) or ``-1`` (open).
            ``(None, None)`` when the user presses 'q' (episode reset signal).
        """
        if self._reset:
            return None, None

        curr_offsets = self._read_offsets()   # degrees relative to zero

        # ── Per-frame delta in radians, scaled to JOINT_POSITION action space ─
        # JOINT_POSITION controller: action × output_max (0.05 rad) = position delta.
        # sensitivity = 1/output_max = 20 → 1° physical maps to 1° simulation.
        delta_rad = [
            (curr_offsets[i] - self._prev_offsets[i]) 
            * (np.pi / 180.0) 
            * self.sensitivity 
            * (3 if i == 6 else 1)
            for i in range(7)  # servos 0-6 are the 7 arm joints
        ]

        # ── 7-DOF Panda: direct 1:1 servo-to-joint mapping ───────────────────
        #  U-ARM has 7 arm servos matching Panda's 7 joints.
        #  No joint insertion or reordering is needed.
        arm_delta = list(delta_rad)       # 7 elements: [Δs0…Δs6]
        # arm_delta.insert(2, 0.0)        # (was for 6-DOF arm only)
        action = np.array(arm_delta, dtype=np.float64)

        # Sign and order corrections.
        action[3] = -action[3]          # joint 3 is physically reversed

        # Clip to the normalised [-1, 1] input range expected by JOINT_POSITION.
        # action = np.clip(action, -1.0, 1.0)

        # ── Gripper command (delta, same convention as arm joints) ──────────
        gripper = (
            (curr_offsets[7] - self._prev_offsets[7])
            * self.sensitivity * 3
        )
        # gripper = float(np.clip(gripper_delta_rad, -1.0, 1.0))
        gripper = -gripper

        # Append gripper commands.  PandaGripper has dof=2 (both fingers coupled).
        gripper_dof = robot.gripper.dof if robot is not None else 2
        action = np.concatenate([action, [gripper] * gripper_dof])

        grasp = 1 if gripper > 0.0 else -1

        self._prev_offsets = curr_offsets
        
        return action, grasp

    # ──────────────────────────────────────────────────────────────────────────
    #  Serial communication (inlined from uarm.py)
    # ──────────────────────────────────────────────────────────────────────────

    def _send_command(self, cmd: str) -> str:
        """Send an ASCII command and return the response string."""
        self._ser.write(cmd.encode("ascii"))
        time.sleep(self._SERIAL_DELAY)
        return self._ser.read_all().decode("ascii", errors="ignore")

    def _pwm_to_angle(self, response_str: str, servo_num: int) -> float:
        """Parse a PWM response string and convert to angle in degrees.

        Returns ``None`` if the response cannot be parsed.
        """
        pattern = f"#{servo_num:03d}P(\\d{{4}})"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        return (
            (pwm_val - self._PWM_MIN)
            / (self._PWM_MAX - self._PWM_MIN)
            * self._ANGLE_RANGE_DEG
        )

    def _init_servos(self) -> None:
        """Send initialisation commands and record the zero-position angles."""
        self._send_command("#000PVER!")
        for i in range(8):           # 7 arm joints (0-6) + gripper (7)
            self._send_command("#000PCSK!")
            self._send_command(f"#{i:03d}PULK!")
            response = self._send_command(f"#{i:03d}PRAD!")
            angle = self._pwm_to_angle(response.strip(), i)
            self._zero_angles[i] = angle if angle is not None else 0.0
        print(
            f"[UArmDevice] Zero calibration complete.  "
            f"Angles: {np.round(self._zero_angles, 2)}"
        )

    def _read_offsets(self) -> list:
        """Query all 7 servos and return their angle offsets from zero (degrees).

        On a failed read for servo *i*, the last known offset is reused so the
        arm does not jump due to a transient serial error.
        """
        offsets = []
        for i in range(8):
            response = self._send_command(f"#{i:03d}PRAD!")
            angle = self._pwm_to_angle(response.strip(), i)
            if angle is not None:
                offsets.append(angle - self._zero_angles[i])
            else:
                # Fall back to last known value to avoid sudden jerks
                offsets.append(self._prev_offsets[i])
                print(
                    f"[UArmDevice] WARNING: servo {i} read failed; "
                    f"holding last value ({self._prev_offsets[i]:.2f}°)."
                )
        return offsets

    # ──────────────────────────────────────────────────────────────────────────
    #  Gripper mapping
    # ──────────────────────────────────────────────────────────────────────────

    def _angle_to_gripper(
        self, angle_rad: float, pos_min: float, pos_max: float
    ) -> float:
        """Map servo 6 absolute angle (radians) to a gripper position in [pos_min, pos_max].

        Convention (matches teleop_sim.py):
            angle_rad = 0             →  pos_max  (gripper closed)
            angle_rad = 1.5π          →  pos_min  (gripper fully open)
        """
        ratio = max(0.0, 1.0 - angle_rad / self._GRIPPER_SERVO_RANGE_RAD)
        position = pos_min + (pos_max - pos_min) * ratio
        return float(np.clip(position, pos_min, pos_max))

    # ──────────────────────────────────────────────────────────────────────────
    #  Keyboard handler
    # ──────────────────────────────────────────────────────────────────────────

    def _on_key_press(self, key) -> None:
        """Set the reset flag when the user presses 'q'."""
        try:
            if hasattr(key, "char") and key.char == "q":
                self._reset = True
                print("\n[UArmDevice] 'q' pressed — ending episode.")
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        try:
            if hasattr(self, "_kb_listener") and self._kb_listener.is_alive():
                self._kb_listener.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_ser") and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
