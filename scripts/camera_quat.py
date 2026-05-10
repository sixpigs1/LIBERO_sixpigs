#!/usr/bin/env python3
"""
Compute the MuJoCo camera quaternion for a given position and look-at point.

The quaternion is in (w, x, y, z) order, matching the MuJoCo XML `quat` attribute.
MuJoCo camera convention: camera looks along its -Z axis; +Y is up.

Usage
-----
  # Camera at (-0.2, 0.8, 1.6) looking at (0.35, 0.0, 0.9):
  python scripts/camera_quat.py -0.2 0.8 1.6

  # Custom look-at target:
  python scripts/camera_quat.py -0.2 0.8 1.6 --target 0.4 0.1 0.85

  # Verify existing cameras (cross-check known values):
  python scripts/camera_quat.py 0.5 0.0 1.35           # agentview
  python scripts/camera_quat.py 1.0 0.0 1.45           # frontview

XML snippet (copy-paste the printed line):
  <camera mode="fixed" name="mycam" pos="..." quat="..." />
"""

import argparse
import sys
import numpy as np


# Default scene look-at target (approx. table-top centre, z = table height)
_DEFAULT_TARGET = (0.35, 0.0, 0.9)


def lookat_quat(cam_pos, target, world_up=None):
    """Return the MuJoCo camera quaternion (w, x, y, z).

    Verified against the agentview camera shipped with LIBERO:
        lookat_quat([0.5, 0, 1.35], [0.05, 0, 0.9])
        → [0.6533, 0.2706, 0.2706, 0.6533]   ✓  (xml: 0.653 0.271 0.271 0.653)

    MuJoCo camera frame:
        col 0 of R  = camera right  (+X)
        col 1 of R  = camera up     (+Y)
        col 2 of R  = backward       (-Z, camera looks along -Z)

    Args:
        cam_pos  : (3,) camera position in world frame.
        target   : (3,) point the camera looks at.
        world_up : (3,) global up direction (default [0, 0, 1]).

    Returns:
        np.ndarray of shape (4,): quaternion [w, x, y, z], positive-w convention.
    """
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    cam_pos = np.asarray(cam_pos, dtype=float)
    target  = np.asarray(target,  dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    forward = target - cam_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-10:
        raise ValueError("cam_pos and target are the same point.")
    forward /= norm

    # Gram–Schmidt: remove any forward component from world_up to get camera right
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise ValueError(
            "forward direction is parallel to world_up; cannot determine camera right."
        )
    right /= right_norm

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    backward = -forward

    # Rotation matrix: columns = [right, up, backward]
    R = np.column_stack([right, up, backward])

    # Rotation matrix → quaternion (Shepperd method, positive-w convention)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q  # canonical positive-w
    return q


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Camera position in world frame.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("TX", "TY", "TZ"),
        default=list(_DEFAULT_TARGET),
        help=(
            "Look-at target in world frame.  "
            f"Default: {_DEFAULT_TARGET} (approx. LIBERO table-top centre)."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mycam",
        help="Camera name for the XML snippet (default: mycam).",
    )
    args = parser.parse_args()

    pos    = args.pos
    target = args.target
    q      = lookat_quat(pos, target)

    px, py, pz = pos
    qw, qx, qy, qz = q

    print()
    print(f"  Camera position : ({px:.4f}, {py:.4f}, {pz:.4f})")
    print(f"  Look-at target  : ({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f})")
    print()
    print(f"  Quaternion (w x y z): {qw:.4f}  {qx:.4f}  {qy:.4f}  {qz:.4f}")
    print()
    print("  XML snippet:")
    print(
        f'    <camera mode="fixed" name="{args.name}" '
        f'pos="{px} {py} {pz}" '
        f'quat="{qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f}" />'
    )
    print()


if __name__ == "__main__":
    main()
