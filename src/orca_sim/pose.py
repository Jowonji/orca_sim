from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import gymnasium as gym
import numpy as np

Side = Literal["left", "right"]
Units = Literal["rad", "deg"]

_FINGER_ABBREV = {
    "t": "thumb",
    "i": "index",
    "m": "middle",
    "r": "ring",
    "p": "pinky",
}
_FINGER_FULL = {full: abbrev for abbrev, full in _FINGER_ABBREV.items()}

# orca_core v2 names for the abbreviated MJCF joints (right_t-pip, …).
_CORE_FROM_ABBREV = {
    "wrist": "wrist",
    "t-cmc": "thumb_cmc",
    "t-abd": "thumb_abd",
    "t-mcp": "thumb_mcp",
    "t-pip": "thumb_dip",
    "i-abd": "index_abd",
    "i-mcp": "index_mcp",
    "i-pip": "index_pip",
    "m-abd": "middle_abd",
    "m-mcp": "middle_mcp",
    "m-pip": "middle_pip",
    "r-abd": "ring_abd",
    "r-mcp": "ring_mcp",
    "r-pip": "ring_pip",
    "p-abd": "pinky_abd",
    "p-mcp": "pinky_mcp",
    "p-pip": "pinky_pip",
}


def _split_side(name: str) -> tuple[Side | None, str]:
    if name.startswith("right_"):
        return "right", name[6:]
    if name.startswith("left_"):
        return "left", name[5:]
    return None, name


def _name_variants(name: str) -> set[str]:
    raw = name.strip()
    lowered = raw.lower()
    return {
        raw,
        lowered,
        raw.replace("-", "_"),
        raw.replace("_", "-"),
        lowered.replace("-", "_"),
        lowered.replace("_", "-"),
    }


def _local_forms(local: str) -> set[str]:
    """Expand a side-stripped joint token into hyphen/underscore aliases."""
    forms = _name_variants(local)
    token = local.replace("_", "-")
    if "-" not in token:
        return forms

    finger, dof = token.split("-", 1)
    if finger in _FINGER_ABBREV:
        full = _FINGER_ABBREV[finger]
        forms.update(_name_variants(f"{full}_{dof}"))
        forms.update(_name_variants(f"{finger}-{dof}"))
    elif finger in _FINGER_FULL:
        abbrev = _FINGER_FULL[finger]
        forms.update(_name_variants(f"{finger}_{dof}"))
        forms.update(_name_variants(f"{abbrev}-{dof}"))
    return forms


def _core_name(local: str) -> str:
    hyphen = local.replace("_", "-")
    if hyphen in _CORE_FROM_ABBREV:
        return _CORE_FROM_ABBREV[hyphen]
    return local.replace("-", "_")


def _actuator_joint_name(model: Any, actuator_id: int) -> str:
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    return model.joint(joint_id).name


class JointNameError(KeyError):
    """Raised when a pose dict uses an unknown or ambiguous joint name."""


class JointPoseMapper:
    """Map orca_core / MJCF joint names onto actuator indices."""

    def __init__(self, model: Any) -> None:
        self.canonical_names: list[str] = []
        self.preferred_names: list[str] = []
        self._index: dict[str, int] = {}
        self._ambiguous: dict[str, tuple[str, ...]] = {}

        claimed_core: dict[str, int] = {}
        records: list[tuple[int, str, Side | None, str, set[str]]] = []
        for actuator_id in range(model.nu):
            canonical = _actuator_joint_name(model, actuator_id)
            side, local = _split_side(canonical)
            core = _core_name(local)
            aliases = _local_forms(local)
            aliases.update(_name_variants(canonical))
            aliases.update(_name_variants(core))
            if side is not None:
                aliases.update(_name_variants(f"{side}_{core}"))
                for form in list(_local_forms(local)):
                    aliases.update(_name_variants(f"{side}_{form}"))
                    aliases.update(_name_variants(f"{side}-{form}"))
            records.append((actuator_id, canonical, side, core, aliases))
            self.canonical_names.append(canonical)
            claimed_core[core] = claimed_core.get(core, 0) + 1

        alias_owners: dict[str, list[int]] = {}
        for actuator_id, _canonical, _side, _core, aliases in records:
            for alias in aliases:
                alias_owners.setdefault(alias, [])
                if actuator_id not in alias_owners[alias]:
                    alias_owners[alias].append(actuator_id)

        for alias, owners in alias_owners.items():
            if len(owners) == 1:
                self._index[alias] = owners[0]
            else:
                self._ambiguous[alias] = tuple(self.canonical_names[i] for i in owners)

        for actuator_id, canonical, side, core, _aliases in records:
            if claimed_core[core] == 1:
                self.preferred_names.append(core)
            elif side is not None:
                self.preferred_names.append(f"{side}_{core}")
            else:
                self.preferred_names.append(canonical)

        self._qpos_indices = np.empty(model.nu, dtype=np.int32)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            self._qpos_indices[actuator_id] = int(model.jnt_qposadr[joint_id])

    @property
    def joint_names(self) -> list[str]:
        return list(self.preferred_names)

    def resolve(self, name: str) -> int:
        if name in self._index:
            return self._index[name]
        for variant in _name_variants(name):
            if variant in self._index:
                return self._index[variant]
            if variant in self._ambiguous:
                options = ", ".join(self._ambiguous[variant])
                raise JointNameError(
                    f"Joint '{name}' is ambiguous. Use one of: {options}."
                )
        available = ", ".join(self.preferred_names)
        raise JointNameError(
            f"Unknown joint '{name}'. Available names: {available}."
        )

    def pose_to_action(
        self,
        pose: Mapping[str, float],
        *,
        base: np.ndarray | None = None,
        fill: float = 0.0,
        units: Units = "rad",
    ) -> np.ndarray:
        if units not in {"rad", "deg"}:
            raise ValueError("units must be 'rad' or 'deg'.")

        n = len(self.canonical_names)
        if base is None:
            action = np.full(n, fill, dtype=np.float32)
        else:
            action = np.asarray(base, dtype=np.float32).copy()
            if action.shape != (n,):
                raise ValueError(f"Expected base shape {(n,)}, got {action.shape}")

        scale = np.pi / 180.0 if units == "deg" else 1.0
        for joint_name, value in pose.items():
            action[self.resolve(joint_name)] = float(value) * scale
        return action

    def action_to_pose(
        self,
        action: np.ndarray,
        *,
        units: Units = "rad",
    ) -> dict[str, float]:
        if units not in {"rad", "deg"}:
            raise ValueError("units must be 'rad' or 'deg'.")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (len(self.preferred_names),):
            raise ValueError(
                f"Expected action shape {(len(self.preferred_names),)}, got {values.shape}"
            )
        scale = 180.0 / np.pi if units == "deg" else 1.0
        return {
            name: float(value) * scale
            for name, value in zip(self.preferred_names, values, strict=True)
        }

    def qpos_to_pose(
        self,
        qpos: np.ndarray,
        *,
        units: Units = "rad",
    ) -> dict[str, float]:
        return self.action_to_pose(np.asarray(qpos)[self._qpos_indices], units=units)


class JointPoseWrapper(gym.Wrapper):
    """Gymnasium wrapper that accepts `{joint_name: angle}` poses.

    Unspecified joints stay at ``fill`` (0 by default). ``step`` still accepts
    a raw action array. Names follow orca_core (``index_mcp``, ``thumb_dip``)
    and also accept MJCF names (``right_i-mcp``).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        units: Units = "rad",
        fill: float = 0.0,
    ) -> None:
        super().__init__(env)
        if units not in {"rad", "deg"}:
            raise ValueError("units must be 'rad' or 'deg'.")
        self.units: Units = units
        self.fill = float(fill)
        self.mapper = JointPoseMapper(env.unwrapped.model)

    @property
    def joint_names(self) -> list[str]:
        return self.mapper.joint_names

    def pose_to_action(
        self,
        pose: Mapping[str, float],
        *,
        base: np.ndarray | Literal["current"] | None = None,
        fill: float | None = None,
        units: Units | None = None,
    ) -> np.ndarray:
        resolved_base: np.ndarray | None
        if isinstance(base, str):
            if base != "current":
                raise ValueError("base must be an array, 'current', or None.")
            resolved_base = np.asarray(self.unwrapped.data.ctrl, dtype=np.float32)
        else:
            resolved_base = base
        return self.mapper.pose_to_action(
            pose,
            base=resolved_base,
            fill=self.fill if fill is None else fill,
            units=self.units if units is None else units,
        )

    def current_pose(
        self,
        *,
        source: Literal["ctrl", "qpos"] = "ctrl",
        units: Units | None = None,
    ) -> dict[str, float]:
        units = self.units if units is None else units
        if source == "ctrl":
            return self.mapper.action_to_pose(self.unwrapped.data.ctrl, units=units)
        if source == "qpos":
            return self.mapper.qpos_to_pose(self.unwrapped.data.qpos, units=units)
        raise ValueError("source must be 'ctrl' or 'qpos'.")

    def step(
        self, action: np.ndarray | Mapping[str, float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, Mapping):
            action = self.pose_to_action(action)
        return self.env.step(action)
