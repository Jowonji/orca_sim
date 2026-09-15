"""Render a saved pose offscreen and write it out as a GIF plus an H.264 mp4.

Mirrors what ``play_pose.py`` shows in the MuJoCo viewer, but renders to an
offscreen buffer so the result can be saved as a file. The mp4 is H.264 so it
plays natively in PowerPoint and Windows Media Player.

    python capture_pose_gif.py saved_poses/pose_002.json -o out/pose_002.gif
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from orca_sim import JointPoseWrapper, OrcaHandRight
from play_pose import load_pose

FPS = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose", type=Path, help="Pose JSON, e.g. saved_poses/pose_002.json")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output .gif path")
    parser.add_argument("--settle", type=float, default=0.7, help="Seconds at the rest pose first.")
    parser.add_argument("--hold", type=float, default=2.3, help="Seconds holding the target pose.")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    joints, units, version = load_pose(args.pose)
    env = JointPoseWrapper(OrcaHandRight(render_mode=None, version=version), units=units)
    env.reset()

    model = env.unwrapped.model
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)

    frames: list[np.ndarray] = []

    def capture() -> None:
        renderer.update_scene(env.unwrapped.data, camera=camera)
        frames.append(np.asarray(renderer.render(), dtype=np.uint8))

    for _ in range(max(1, int(args.settle * FPS))):
        env.step({})
        capture()
    for _ in range(max(1, int(args.hold * FPS))):
        env.step(joints)
        capture()

    renderer.close()
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mp4_out = args.out.with_suffix(".mp4")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(tmp / f"{i:05d}.png")
        palette = tmp / "palette.png"
        common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                  "-framerate", str(FPS), "-i", str(tmp / "%05d.png")]
        subprocess.run(common + ["-vf", "palettegen=stats_mode=diff", str(palette)], check=True)
        subprocess.run(
            common + ["-i", str(palette),
                      "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", str(args.out)],
            check=True,
        )
        # yuv420p + even dimensions keep this playable in PowerPoint.
        subprocess.run(
            common + ["-c:v", "libx264", "-preset", "slow", "-crf", "20",
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_out)],
            check=True,
        )

    for path in (args.out, mp4_out):
        size_kb = path.stat().st_size / 1024
        print(f"{path}  {len(frames)} frames, {len(frames) / FPS:.1f}s, {size_kb:.0f} KB")
    print("target pose:", {k: v for k, v in joints.items() if v})


if __name__ == "__main__":
    main()
