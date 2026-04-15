# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
    "to_relative_actions_with_pose_specs",
    "to_absolute_actions_with_pose_specs",
]


@dataclass(frozen=True)
class _PoseActionSpec:
    position_indices: tuple[int, int, int]
    quaternion_indices: tuple[int, int, int, int]


_QUATERNION_COMPONENT_PATTERNS = (
    re.compile(r"^(?P<prefix>.*?)(?:\.)?(?:quat|quaternion|orientation)\.(?P<component>[wxyz])$"),
    re.compile(r"^(?P<prefix>.*?)(?P<component>[wxyz])\.(?:quat|quaternion|orientation)$"),
    re.compile(r"^(?P<prefix>.*?)(?:\.)?q(?P<component>[wxyz])$"),
)
_POSITION_COMPONENT_PATTERNS = (
    re.compile(r"^(?P<prefix>.*?)(?:\.)?(?:pos|position)\.(?P<component>[xyz])$"),
    re.compile(r"^(?P<prefix>.*?)(?P<component>[xyz])\.(?:pos|position)$"),
)
_AXIS_COMPONENT_PATTERNS = (
    re.compile(r"^(?P<prefix>.+)\.(?P<component>[xyz])$"),
    re.compile(r"^(?P<component>[xyz])$"),
)


def _extract_component(name: str, patterns: Sequence[re.Pattern[str]]) -> tuple[str, str] | None:
    normalized_name = re.sub(r"[/_]", ".", name.lower())
    for pattern in patterns:
        match = pattern.match(normalized_name)
        if match is None:
            continue
        prefix = match.groupdict().get("prefix", "").strip(".")
        component = match.group("component")
        return prefix, component
    return None


def _normalize_quaternion_wxyz(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(eps)


def _quat_conjugate_wxyz(quaternion: Tensor) -> Tensor:
    conjugate = quaternion.clone()
    conjugate[..., 1:] = -conjugate[..., 1:]
    return conjugate


def _quat_multiply_wxyz(lhs: Tensor, rhs: Tensor) -> Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_apply_wxyz(quaternion: Tensor, vectors: Tensor) -> Tensor:
    q_xyz = quaternion[..., 1:]
    q_w = quaternion[..., :1]
    t = 2.0 * torch.linalg.cross(q_xyz, vectors, dim=-1)
    return vectors + q_w * t + torch.linalg.cross(q_xyz, t, dim=-1)


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


def to_relative_actions_with_pose_specs(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    pose_specs: Sequence[_PoseActionSpec],
) -> Tensor:
    if not pose_specs:
        return to_relative_actions(actions, state, mask)

    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    absolute_actions = actions
    relative_actions = to_relative_actions(actions, state, mask)

    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=actions.device)
    dims = mask_t.shape[0]
    state_dim = state.shape[-1]
    mask_list = mask_t.tolist()

    for spec in pose_specs:
        pose_indices = spec.position_indices + spec.quaternion_indices
        if any(index >= dims or index >= state_dim for index in pose_indices):
            continue
        if not all(mask_list[index] for index in spec.quaternion_indices):
            continue

        state_quat = _normalize_quaternion_wxyz(state[..., list(spec.quaternion_indices)])
        action_quat = _normalize_quaternion_wxyz(absolute_actions[..., list(spec.quaternion_indices)])
        if relative_actions.ndim == 3:
            state_quat = state_quat.unsqueeze(-2)

        # q and -q encode the same orientation. Align to avoid discontinuous jumps.
        dot = (action_quat * state_quat).sum(dim=-1, keepdim=True)
        action_quat = torch.where(dot < 0.0, -action_quat, action_quat)

        relative_quat = _quat_multiply_wxyz(_quat_conjugate_wxyz(state_quat), action_quat)
        relative_actions[..., list(spec.quaternion_indices)] = _normalize_quaternion_wxyz(relative_quat)

        if all(mask_list[index] for index in spec.position_indices):
            state_pos = state[..., list(spec.position_indices)]
            action_pos = absolute_actions[..., list(spec.position_indices)]
            if relative_actions.ndim == 3:
                state_pos = state_pos.unsqueeze(-2)
            relative_pos = _quat_apply_wxyz(_quat_conjugate_wxyz(state_quat), action_pos - state_pos)
            relative_actions[..., list(spec.position_indices)] = relative_pos

    return relative_actions


def to_absolute_actions_with_pose_specs(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    pose_specs: Sequence[_PoseActionSpec],
) -> Tensor:
    if not pose_specs:
        return to_absolute_actions(actions, state, mask)

    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    relative_actions = actions
    absolute_actions = to_absolute_actions(actions, state, mask)

    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=actions.device)
    dims = mask_t.shape[0]
    state_dim = state.shape[-1]
    mask_list = mask_t.tolist()

    for spec in pose_specs:
        pose_indices = spec.position_indices + spec.quaternion_indices
        if any(index >= dims or index >= state_dim for index in pose_indices):
            continue
        if not all(mask_list[index] for index in spec.quaternion_indices):
            continue

        state_quat = _normalize_quaternion_wxyz(state[..., list(spec.quaternion_indices)])
        relative_quat = _normalize_quaternion_wxyz(relative_actions[..., list(spec.quaternion_indices)])
        if absolute_actions.ndim == 3:
            state_quat = state_quat.unsqueeze(-2)

        absolute_quat = _quat_multiply_wxyz(state_quat, relative_quat)
        absolute_actions[..., list(spec.quaternion_indices)] = _normalize_quaternion_wxyz(absolute_quat)

        if all(mask_list[index] for index in spec.position_indices):
            state_pos = state[..., list(spec.position_indices)]
            relative_pos = relative_actions[..., list(spec.position_indices)]
            if absolute_actions.ndim == 3:
                state_pos = state_pos.unsqueeze(-2)
            absolute_pos = state_pos + _quat_apply_wxyz(state_quat, relative_pos)
            absolute_actions[..., list(spec.position_indices)] = absolute_pos

    return absolute_actions


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def _build_pose_specs(self, action_dim: int) -> list[_PoseActionSpec]:
        if self.action_names is None:
            return []

        action_names = [str(name) for name in self.action_names[:action_dim]]
        quat_components_by_prefix: dict[str, dict[str, int]] = {}
        pos_components_by_prefix: dict[str, dict[str, int]] = {}

        for index, action_name in enumerate(action_names):
            quat_component = _extract_component(action_name, _QUATERNION_COMPONENT_PATTERNS)
            if quat_component is not None:
                prefix, component = quat_component
                quat_components_by_prefix.setdefault(prefix, {})[component] = index
                continue

            pos_component = _extract_component(action_name, _POSITION_COMPONENT_PATTERNS)
            if pos_component is not None:
                prefix, component = pos_component
                pos_components_by_prefix.setdefault(prefix, {})[component] = index

        # Fallback for common "<prefix>.x/y/z" position naming.
        if quat_components_by_prefix:
            for index, action_name in enumerate(action_names):
                axis_component = _extract_component(action_name, _AXIS_COMPONENT_PATTERNS)
                if axis_component is None:
                    continue
                prefix, component = axis_component
                if prefix not in quat_components_by_prefix:
                    continue
                pos_components_by_prefix.setdefault(prefix, {}).setdefault(component, index)

        pose_specs = []
        for prefix, quat_components in quat_components_by_prefix.items():
            if not {"w", "x", "y", "z"}.issubset(quat_components):
                continue
            pos_components = pos_components_by_prefix.get(prefix, {})
            if not {"x", "y", "z"}.issubset(pos_components):
                continue
            pose_specs.append(
                _PoseActionSpec(
                    position_indices=(pos_components["x"], pos_components["y"], pos_components["z"]),
                    quaternion_indices=(
                        quat_components["w"],
                        quat_components["x"],
                        quat_components["y"],
                        quat_components["z"],
                    ),
                )
            )
            print(" [_PoseActionSpec] ", pose_specs[-1])

        pose_specs.sort(key=lambda spec: min(spec.position_indices + spec.quaternion_indices))
        return pose_specs

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # Always cache state for the paired AbsoluteActionsProcessorStep
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        pose_specs = self._build_pose_specs(action.shape[-1])
        if pose_specs:
            new_transition[TransitionKey.ACTION] = to_relative_actions_with_pose_specs(
                action, state, mask, pose_specs
            )
        else:
            new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        if self.relative_step._last_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        pose_specs = self.relative_step._build_pose_specs(action.shape[-1])
        if pose_specs:
            new_transition[TransitionKey.ACTION] = to_absolute_actions_with_pose_specs(
                action, self.relative_step._last_state, mask, pose_specs
            )
        else:
            new_transition[TransitionKey.ACTION] = to_absolute_actions(
                action, self.relative_step._last_state, mask
            )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
