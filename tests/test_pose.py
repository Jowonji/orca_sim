import numpy as np
import pytest

from orca_sim import (
    JointNameError,
    JointPoseWrapper,
    OrcaHandCombined,
    OrcaHandLeft,
    OrcaHandRight,
)


def test_right_v2_accepts_orca_core_names() -> None:
    env = JointPoseWrapper(OrcaHandRight(version="v2"))
    try:
        env.reset()
        action = env.pose_to_action({"index_mcp": 1.0, "index_pip": 1.2})

        assert env.joint_names[11] == "index_mcp"
        assert env.joint_names[12] == "index_pip"
        assert action[11] == pytest.approx(1.0)
        assert action[12] == pytest.approx(1.2)
        assert action[0] == pytest.approx(0.0)

        env.step({"index_mcp": 1.0, "index_pip": 1.2, "thumb_dip": 0.5})
        pose = env.current_pose()
        assert pose["index_mcp"] == pytest.approx(1.0)
        assert pose["index_pip"] == pytest.approx(1.2)
        assert pose["thumb_dip"] == pytest.approx(0.5)
    finally:
        env.close()


def test_right_v2_accepts_mjcf_and_side_prefixed_names() -> None:
    env = JointPoseWrapper(OrcaHandRight(version="v2"))
    try:
        by_core = env.pose_to_action({"thumb_cmc": 0.2, "wrist": -0.3})
        by_mjcf = env.pose_to_action({"right_t-cmc": 0.2, "right_wrist": -0.3})
        by_side = env.pose_to_action({"right_thumb_cmc": 0.2, "right_wrist": -0.3})
        np.testing.assert_allclose(by_core, by_mjcf)
        np.testing.assert_allclose(by_core, by_side)
    finally:
        env.close()


def test_right_v1_keeps_thumb_pip_and_thumb_dip_distinct() -> None:
    env = JointPoseWrapper(OrcaHandRight(version="v1"))
    try:
        action = env.pose_to_action({"thumb_pip": 0.4, "thumb_dip": 0.8})
        pose = env.mapper.action_to_pose(action)
        assert pose["thumb_pip"] == pytest.approx(0.4)
        assert pose["thumb_dip"] == pytest.approx(0.8)
        assert pose["thumb_pip"] != pose["thumb_dip"]
    finally:
        env.close()


def test_degrees_are_converted_to_radians() -> None:
    env = JointPoseWrapper(OrcaHandLeft(version="v2"), units="deg")
    try:
        action = env.pose_to_action({"index_mcp": 90.0})
        assert action[env.joint_names.index("index_mcp")] == pytest.approx(np.pi / 2)
        env.reset()
        env.step({"index_mcp": 45.0})
        assert env.current_pose()["index_mcp"] == pytest.approx(45.0)
        assert env.current_pose(units="rad")["index_mcp"] == pytest.approx(np.pi / 4)
    finally:
        env.close()


def test_combined_requires_side_prefix() -> None:
    env = JointPoseWrapper(OrcaHandCombined(version="v2"))
    try:
        with pytest.raises(JointNameError, match="ambiguous"):
            env.pose_to_action({"index_mcp": 1.0})

        action = env.pose_to_action(
            {"right_index_mcp": 1.0, "left_index_mcp": 0.5}
        )
        pose = env.mapper.action_to_pose(action)
        assert pose["right_index_mcp"] == pytest.approx(1.0)
        assert pose["left_index_mcp"] == pytest.approx(0.5)
    finally:
        env.close()


def test_unknown_joint_lists_available_names() -> None:
    env = JointPoseWrapper(OrcaHandRight(version="v2"))
    try:
        with pytest.raises(JointNameError, match="index_mcp"):
            env.pose_to_action({"not_a_joint": 1.0})
    finally:
        env.close()


def test_partial_pose_can_merge_onto_current_ctrl() -> None:
    env = JointPoseWrapper(OrcaHandRight(version="v2"))
    try:
        env.reset()
        env.step({"index_mcp": 0.7})
        action = env.pose_to_action({"index_pip": 0.4}, base="current")
        assert action[env.joint_names.index("index_mcp")] == pytest.approx(0.7)
        assert action[env.joint_names.index("index_pip")] == pytest.approx(0.4)
    finally:
        env.close()


def test_raw_action_array_still_works() -> None:
    env = JointPoseWrapper(OrcaHandRight(version="v2"))
    try:
        env.reset()
        zeros = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(zeros)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert terminated is False
        assert truncated is False
        assert info == {}
    finally:
        env.close()
