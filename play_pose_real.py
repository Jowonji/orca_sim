"""Replay a saved pose JSON on the real ORCA hand via orca_core.

The JSON from interactive_pose.py already uses orca_core joint names in degrees,
so the same file can drive MuJoCo (play_pose.py) and the physical hand.

This does **not** prove the motions are physically identical. It only sends the
same joint-space command. Tendons, calibration, and dynamics still differ.

Examples:
    python play_pose_real.py --mock
    python play_pose_real.py --real
    python play_pose_real.py --real --sim saved_poses/pose_001.json
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from play_pose import DEFAULT_POSE, load_pose

REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent


def _ensure_orca_core() -> None:
    try:
        import orca_core  # noqa: F401
        return
    except ImportError:
        core_root = WORKSPACE / "orca_core"
        if core_root.is_dir():
            sys.path.insert(0, str(core_root))
        import orca_core  # noqa: F401


def to_degrees(joints: dict[str, float], units: str) -> dict[str, float]:
    if units == "deg":
        return dict(joints)
    scale = 180.0 / math.pi
    return {name: value * scale for name, value in joints.items()}


def print_compare(
    commanded: dict[str, float],
    hardware: dict[str, float] | None,
    sim: dict[str, float] | None,
) -> None:
    names = list(commanded)
    print("\njoint            cmd_deg    hw_deg    sim_deg")
    print("-" * 52)
    for name in names:
        cmd = commanded[name]
        hw = hardware.get(name) if hardware else None
        sm = sim.get(name) if sim else None
        hw_s = f"{hw:8.1f}" if hw is not None else "     n/a"
        sm_s = f"{sm:8.1f}" if sm is not None else "     n/a"
        print(f"{name:<16} {cmd:8.1f} {hw_s} {sm_s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="저장된 pose JSON을 orca_core로 실제 손(또는 mock)에 재생합니다."
    )
    parser.add_argument(
        "pose",
        nargs="?",
        type=Path,
        default=DEFAULT_POSE,
        help="saved_poses/pose_001.json 같은 pose 파일.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="실제 손을 움직입니다. 이 플래그가 없으면 하드웨어에 명령을 보내지 않습니다.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="시리얼 없이 orca_core MockOrcaHand로 파이프라인만 확인합니다.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="같은 pose를 MuJoCo 창에도 같이 넣습니다.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="orca_core config.yaml 경로. 생략하면 load_hand()가 모델을 고릅니다.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=4.0,
        help="목표 pose를 유지하는 시간(초).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="실제 손 보간 스텝 수. 클수록 천천히 갑니다.",
    )
    args = parser.parse_args()

    if not args.real and not args.mock and not args.sim:
        raise SystemExit(
            "무엇을 재생할지 고르세요: --real (실제 손), --mock (orca_core mock), "
            "또는 --sim (시뮬만)."
        )

    pose_path = args.pose.expanduser()
    if not pose_path.is_absolute():
        pose_path = (Path.cwd() / pose_path).resolve()
    if not pose_path.is_file():
        raise SystemExit(f"pose 파일이 없습니다: {pose_path}")

    joints, units, version = load_pose(pose_path)
    joints_deg = to_degrees(joints, units)
    print(f"pose={pose_path}")
    print(f"version={version} units={units} -> hardware uses deg")

    env = None
    if args.sim:
        from orca_sim import JointPoseWrapper, OrcaHandRight

        env = JointPoseWrapper(
            OrcaHandRight(render_mode="human", version=version),
            units="deg",
        )
        env.reset()

    hand = None
    if args.real or args.mock:
        if args.real and args.mock:
            raise SystemExit("--real 과 --mock 을 같이 쓸 수 없습니다.")
        if args.real:
            print(
                "실제 손이 움직입니다. 충돌할 물건을 치우고, "
                "텐션/캘리브레이션이 끝난 상태인지 확인하세요."
            )
        _ensure_orca_core()
        from orca_core import load_hand
        from orca_core.utils.cli import connect_hand, shutdown_hand

        hand = load_hand(config_path=args.config, mock=args.mock)
        connect_hand(hand)
        hand.init_joints(force_calibrate=args.mock, move_to_neutral=True)
        known = set(hand.config.joint_ids)
        missing = [name for name in joints_deg if name not in known]
        if missing:
            print(f"이 손 config에 없는 관절은 건너뜁니다: {missing}")
        joints_deg = {name: value for name, value in joints_deg.items() if name in known}

    try:
        if env is not None:
            for _ in range(30):
                env.step({})
                time.sleep(1 / 30)

        if hand is not None:
            hand.set_joint_positions(
                joints_deg,
                num_steps=max(1, args.num_steps),
                step_size=0.01,
            )

        hold_frames = max(1, int(args.hold * 30))
        for _ in range(hold_frames):
            if env is not None:
                env.step(joints_deg)
            time.sleep(1 / 30)

        hardware_pose = None
        if hand is not None:
            hardware_pose = {
                name: value
                for name, value in hand.get_joint_position().as_dict().items()
                if value is not None
            }
        sim_pose = None
        if env is not None:
            sim_pose = env.current_pose(source="qpos", units="deg")
        print_compare(joints_deg, hardware_pose, sim_pose)
    finally:
        if env is not None:
            env.close()
        if hand is not None:
            from orca_core.utils.cli import shutdown_hand

            try:
                hand.set_neutral_position()
            except Exception as exc:
                print(f"neutral 복귀 실패: {exc}")
            shutdown_hand(hand)


if __name__ == "__main__":
    main()
