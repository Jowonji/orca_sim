"""Play a saved target pose in the MuJoCo viewer.

Examples:
    python play_pose.py
    python play_pose.py saved_poses/pose_001.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from orca_sim import JointPoseWrapper, OrcaHandRight

REPO = Path(__file__).resolve().parent
DEFAULT_POSE = REPO / "saved_poses" / "pose_001.json"


def load_pose(path: Path) -> tuple[dict[str, float], str, str | None]:
    payload = json.loads(path.read_text())
    if "joints" not in payload:
        raise ValueError(f"{path} 에 'joints' 가 없습니다.")
    units = str(payload.get("units", "deg"))
    if units not in {"deg", "rad"}:
        raise ValueError(f"지원하지 않는 units: {units}")
    version = payload.get("version")
    joints = {name: float(value) for name, value in payload["joints"].items()}
    return joints, units, None if version is None else str(version)


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 target pose를 시뮬레이터에서 재생합니다.")
    parser.add_argument(
        "pose",
        nargs="?",
        type=Path,
        default=DEFAULT_POSE,
        help="saved_poses/pose_001.json 같은 pose 파일. 생략하면 pose_001.json 을 씁니다.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=4.0,
        help="목표 pose를 유지하는 시간(초).",
    )
    args = parser.parse_args()
    pose_path = args.pose.expanduser()
    if not pose_path.is_absolute():
        pose_path = (Path.cwd() / pose_path).resolve()
    if not pose_path.is_file():
        raise SystemExit(f"pose 파일이 없습니다: {pose_path}")

    joints, units, version = load_pose(pose_path)
    env = JointPoseWrapper(
        OrcaHandRight(render_mode="human", version=version),
        units=units,
    )
    env.reset()

    print(f"pose={pose_path}")
    print(f"version={env.unwrapped.version} units={units}")
    print("target=")
    for name, value in joints.items():
        print(f"  {name}: {value:.1f}")

    try:
        for _ in range(30):
            env.step({})
            time.sleep(1 / 30)

        hold_frames = max(1, int(args.hold * 30))
        for _ in range(hold_frames):
            env.step(joints)
            time.sleep(1 / 30)
    finally:
        env.close()


if __name__ == "__main__":
    main()
