"""Interactive pose editor: drive the hand from the MuJoCo window (or the terminal).

The previous version used ``input()``. That only reads the *terminal* after Enter,
and it blocks ``step()`` / ``sync()``, so the viewer looks frozen and keys in the
simulator window do nothing. This script keeps the sim running and binds keys
through the MuJoCo viewer callback (plus non-blocking stdin as a fallback).
"""

from __future__ import annotations

import json
import select
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue

import mujoco
import numpy as np
from mujoco import viewer

from orca_sim import JointPoseWrapper, OrcaHandRight

try:
    import glfw
except ImportError:  # pragma: no cover
    glfw = None

try:
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None
    tty = None


STEP_CHOICES = {1.0, 5.0, 10.0}
POSE_DIR = Path(__file__).resolve().parent / "saved_poses"

JOINT_HELP = {
    "wrist": "손목을 손바닥 쪽으로 꺾기",
    "thumb_cmc": "엄지 뿌리를 손바닥 안쪽으로 돌리기",
    "thumb_abd": "엄지를 다른 손가락 쪽으로 벌리기",
    "thumb_mcp": "엄지 첫마디 접기",
    "thumb_dip": "엄지 끝마디 접기",
    "index_abd": "검지를 옆으로 기울이기 (소지 쪽)",
    "index_mcp": "검지 너클(뿌리) 접기",
    "index_pip": "검지 중간마디 접기",
    "middle_abd": "중지를 옆으로 기울이기 (소지 쪽)",
    "middle_mcp": "중지 너클(뿌리) 접기",
    "middle_pip": "중지 중간마디 접기",
    "ring_abd": "약지를 옆으로 기울이기 (소지 쪽)",
    "ring_mcp": "약지 너클(뿌리) 접기",
    "ring_pip": "약지 중간마디 접기",
    "pinky_abd": "소지를 바깥으로 벌리기",
    "pinky_mcp": "소지 너클(뿌리) 접기",
    "pinky_pip": "소지 중간마디 접기",
}


class TerminalKeys:
    """Read single keypresses from stdin without waiting for Enter."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old: list | None = None
        self._enabled = False

    def __enter__(self) -> TerminalKeys:
        if (
            termios is None
            or tty is None
            or not sys.stdin.isatty()
        ):
            return self
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._enabled = True
        return self

    def __exit__(self, *exc: object) -> None:
        if self._enabled and self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        self._enabled = False

    def read(self) -> list[str]:
        if not self._enabled:
            return []
        chunks: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                break
            chunks.append(sys.stdin.read(1))
        if not chunks:
            return []
        return _decode_keys("".join(chunks))


def _decode_keys(raw: str) -> list[str]:
    keys: list[str] = []
    i = 0
    while i < len(raw):
        if raw.startswith("\x1b[", i) and i + 2 < len(raw):
            token = raw[i + 2]
            keys.append(
                {"A": "inc", "B": "dec", "C": "next", "D": "prev"}.get(token, "")
            )
            i += 3
            continue
        ch = raw[i]
        i += 1
        if ch in {"\x03", "q", "Q"}:
            keys.append("quit")
        elif ch in {"n", "N"}:
            keys.append("next")
        elif ch in {"p", "P"}:
            keys.append("prev")
        elif ch in {"+", "=", "]"}:
            keys.append("inc")
        elif ch in {"-", "_", "["}:
            keys.append("dec")
        elif ch == "1":
            keys.append("step1")
        elif ch == "5":
            keys.append("step5")
        elif ch == "0":
            keys.append("step10")
        elif ch in {"s", "S"}:
            keys.append("save")
        elif ch in {"r", "R"}:
            keys.append("reset")
        elif ch in {"\x1b"}:
            continue
    return [key for key in keys if key]


def _glfw_command(key: int) -> str | None:
    if glfw is None:
        mapping = {
            ord("Q"): "quit",
            ord("N"): "next",
            ord("P"): "prev",
            ord("S"): "save",
            ord("R"): "reset",
            ord("1"): "step1",
            ord("5"): "step5",
            ord("0"): "step10",
            ord("="): "inc",
            ord("-"): "dec",
            ord("]"): "inc",
            ord("["): "dec",
            262: "next",
            263: "prev",
            264: "dec",
            265: "inc",
            333: "dec",
            334: "inc",
        }
        return mapping.get(key)

    mapping = {
        glfw.KEY_Q: "quit",
        glfw.KEY_N: "next",
        glfw.KEY_P: "prev",
        glfw.KEY_S: "save",
        glfw.KEY_R: "reset",
        glfw.KEY_1: "step1",
        glfw.KEY_5: "step5",
        glfw.KEY_0: "step10",
        glfw.KEY_EQUAL: "inc",
        glfw.KEY_MINUS: "dec",
        glfw.KEY_RIGHT_BRACKET: "inc",
        glfw.KEY_LEFT_BRACKET: "dec",
        glfw.KEY_RIGHT: "next",
        glfw.KEY_LEFT: "prev",
        glfw.KEY_UP: "inc",
        glfw.KEY_DOWN: "dec",
        glfw.KEY_KP_ADD: "inc",
        glfw.KEY_KP_SUBTRACT: "dec",
        glfw.KEY_KP_1: "step1",
        glfw.KEY_KP_5: "step5",
        glfw.KEY_KP_0: "step10",
    }
    return mapping.get(key)


def format_pose(pose: dict[str, float], joint_names: list[str]) -> str:
    lines = ["{"]
    for joint in joint_names:
        lines.append(f'    "{joint}": {pose[joint]:.1f},')
    lines.append("}")
    return "\n".join(lines)


def next_pose_path(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in directory.glob("pose_*.json"):
        suffix = path.stem.removeprefix("pose_")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return directory / f"pose_{max(numbers, default=0) + 1:03d}.json"


def save_target_pose(
    pose: dict[str, float],
    joint_names: list[str],
    *,
    version: str,
) -> Path:
    path = next_pose_path(POSE_DIR)
    payload = {
        "units": "deg",
        "version": version,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "joints": {name: round(float(pose[name]), 1) for name in joint_names},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    env = JointPoseWrapper(
        OrcaHandRight(version="v2", render_mode=None),
        units="deg",
    )
    env.reset()

    joint_names = list(env.joint_names)
    pose = {joint: 0.0 for joint in joint_names}
    selected = 0
    step_deg = 5.0
    pending: SimpleQueue[str] = SimpleQueue()
    last_discrete_at = 0.0

    limits = {
        name: (
            float(np.rad2deg(env.unwrapped.action_low[i])),
            float(np.rad2deg(env.unwrapped.action_high[i])),
        )
        for i, name in enumerate(joint_names)
    }

    def queue_command(command: str | None) -> None:
        nonlocal last_discrete_at
        if not command:
            return
        now = time.monotonic()
        if command not in {"inc", "dec"} and now - last_discrete_at < 0.12:
            return
        if command not in {"inc", "dec"}:
            last_discrete_at = now
        pending.put(command)

    def on_key(key: int) -> None:
        queue_command(_glfw_command(key))

    def print_status() -> None:
        joint = joint_names[selected]
        print(
            f"[{selected + 1}/{len(joint_names)}] {joint}  "
            f"{JOINT_HELP.get(joint, '')}  "
            f"{pose[joint]:.1f} deg  (step {step_deg:.0f})"
        )

    def apply_command(command: str) -> str | None:
        nonlocal selected, step_deg, pose
        joint = joint_names[selected]
        lo, hi = limits[joint]
        changed = True

        if command == "quit":
            return "quit"
        if command == "next":
            selected = (selected + 1) % len(joint_names)
        elif command == "prev":
            selected = (selected - 1) % len(joint_names)
        elif command == "inc":
            pose[joint] = float(np.clip(pose[joint] + step_deg, lo, hi))
        elif command == "dec":
            pose[joint] = float(np.clip(pose[joint] - step_deg, lo, hi))
        elif command == "step1":
            step_deg = 1.0
        elif command == "step5":
            step_deg = 5.0
        elif command == "step10":
            step_deg = 10.0
        elif command == "reset":
            pose = {name: 0.0 for name in joint_names}
        elif command == "save":
            path = save_target_pose(
                pose,
                joint_names,
                version=str(env.unwrapped.version),
            )
            print(f"\n저장됨: {path}")
            print(format_pose(pose, joint_names))
            print()
            changed = False
        else:
            changed = False
        if changed:
            print_status()
        return None

    def overlay_texts() -> tuple[int, int, str, str]:
        joint = joint_names[selected]
        lo, hi = limits[joint]
        return (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "joint\nangle\nrom\nstep\nkeys",
            (
                f"{joint}\n"
                f"{pose[joint]:.1f} deg\n"
                f"{lo:.0f} .. {hi:.0f}\n"
                f"{step_deg:.0f} deg\n"
                "n/p  +/-  1 5 0  s  r  q"
            ),
        )

    print(
        """
관절 이름 설명
--------------
wrist       손목을 손바닥 쪽으로 꺾기
thumb_cmc   엄지 뿌리를 손바닥 안쪽으로 돌리기
thumb_abd   엄지를 다른 손가락 쪽으로 벌리기
thumb_mcp   엄지 첫마디 접기
thumb_dip   엄지 끝마디 접기
index_*     검지  /  middle_* 중지  /  ring_* 약지  /  pinky_* 소지
*_abd       손가락을 옆으로 벌리거나 기울이기
*_mcp       너클(손등 쪽 뿌리) 접기
*_pip       중간마디 접기

조작: n/p 관절 선택   +/- 각도   1/5/0 간격   s 파일저장   r 리셋   q 종료
s 를 누르면 saved_poses/pose_001.json 처럼 번호가 올라가며 저장됩니다.
한글 설명은 이 터미널에 나옵니다. (시뮬 창 폰트는 한글을 못 그림)
"""
    )
    print_status()

    model = env.unwrapped.model
    data = env.unwrapped.data
    try:
        with viewer.launch_passive(model, data, key_callback=on_key) as handle:
            mujoco.mjv_defaultFreeCamera(model, handle.cam)
            with TerminalKeys() as terminal:
                while handle.is_running():
                    for command in terminal.read():
                        queue_command(command)

                    while not pending.empty():
                        if apply_command(pending.get_nowait()) == "quit":
                            handle.close()
                            break

                    env.step(pose)
                    handle.set_texts(overlay_texts())
                    handle.sync()
                    time.sleep(1.0 / 60.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
