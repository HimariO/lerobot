#!/usr/bin/env python

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

import time
from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.processor import DataProcessorPipeline, EnvTransition, ProcessorStep, RobotObservation, TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.teleoperators import Teleoperator


AIC_DEFAULT_EE_POSE_ACTION_KEYS = (
    "position.x",
    "position.y",
    "position.z",
    "orientation.x",
    "orientation.y",
    "orientation.z",
    "orientation.w",
)


def get_feature_shape(feature: Any) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if shape is None:
        raise ValueError(f"Feature has no shape: {feature}")
    return tuple(int(v) for v in shape)


def get_feature_names(feature: Any) -> list[str] | None:
    names = getattr(feature, "names", None)
    if names is None and isinstance(feature, dict):
        names = feature.get("names")
    if names is None:
        return None
    return [str(v) for v in names]


def get_feature_type(feature: Any, key: str) -> FeatureType:
    feature_type = getattr(feature, "type", None)
    if feature_type is None and isinstance(feature, dict):
        feature_type = feature.get("type")

    if isinstance(feature_type, FeatureType):
        return feature_type
    if isinstance(feature_type, str):
        return FeatureType(feature_type)

    if key == ACTION:
        return FeatureType.ACTION
    if key.startswith(f"{OBS_IMAGES}."):
        return FeatureType.VISUAL
    if key.startswith(OBS_STATE):
        return FeatureType.STATE
    if key.startswith(OBS_ENV_STATE):
        return FeatureType.ENV

    raise ValueError(f"Could not infer feature type for key '{key}'.")


def resolve_action_key_order(cfg: HILSerlRobotEnvConfig, fallback_action_keys: Sequence[str]) -> list[str]:
    action_feature = cfg.features.get(ACTION)
    if action_feature is not None:
        action_names = get_feature_names(action_feature)
        if action_names:
            return action_names

    fallback = list(fallback_action_keys)
    if fallback:
        return fallback
    return list(AIC_DEFAULT_EE_POSE_ACTION_KEYS)


def get_observation_feature_specs(
    cfg: HILSerlRobotEnvConfig,
) -> list[tuple[str, str, FeatureType, tuple[int, ...], list[str] | None]]:
    observation_specs: list[tuple[str, str, FeatureType, tuple[int, ...], list[str] | None]] = []

    for source_key, feature in cfg.features.items():
        mapped_key = cfg.features_map.get(source_key, source_key)
        feature_type = get_feature_type(feature, mapped_key)
        if feature_type == FeatureType.ACTION:
            continue
        if not mapped_key.startswith("observation."):
            continue

        observation_specs.append(
            (
                str(source_key),
                str(mapped_key),
                feature_type,
                get_feature_shape(feature),
                get_feature_names(feature),
            )
        )

    return observation_specs


@dataclass
class PolicyObservationPreprocessorStep(ProcessorStep):
    policy_preprocessor: DataProcessorPipeline[dict[str, Any], dict[str, Any]]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise ValueError("PolicyObservationPreprocessorStep requires an observation dictionary.")
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = self.policy_preprocessor(observation)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class PolicyActionPostprocessorStep(ProcessorStep):
    policy_postprocessor: DataProcessorPipeline[torch.Tensor, torch.Tensor]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"PolicyActionPostprocessorStep expects tensor action, got {type(action)}")

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = self.policy_postprocessor(action)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


class AICRobotEnv(gym.Env):
    """Robot environment variant that projects observations to the configured AIC schema."""

    def __init__(
        self,
        robot,
        teleop_device: Teleoperator,
        cfg: HILSerlRobotEnvConfig,
        display_cameras: bool = False,
        reset_time_s: float = 2.0,
    ) -> None:
        super().__init__()
        self.robot = robot
        self.teleop_device = teleop_device
        self.cfg = cfg
        self.display_cameras = display_cameras
        self.reset_time_s = reset_time_s

        if not self.robot.is_connected:
            self.robot.connect()

        self.current_step = 0
        self.episode_data = None
        self._raw_joint_positions: dict[str, float] = {}

        fallback_action_keys = list(getattr(self.robot, "action_features", {}).keys())
        self.action_keys = resolve_action_key_order(cfg=cfg, fallback_action_keys=fallback_action_keys)

        self._observation_specs = get_observation_feature_specs(cfg)
        if not self._observation_specs:
            raise ValueError("AICRobotEnv requires observation features in env.features/env.features_map.")

        self._setup_spaces()

    def _setup_spaces(self) -> None:
        observation_spaces: dict[str, gym.spaces.Space] = {}
        for _source_key, mapped_key, feature_type, shape, _names in self._observation_specs:
            if feature_type == FeatureType.VISUAL:
                if len(shape) != 3:
                    raise ValueError(f"Visual feature '{mapped_key}' must be 3D, got shape={shape}.")
                c, h, w = shape
                if shape[-1] in (1, 3):
                    h, w, c = shape
                observation_spaces[mapped_key] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(c, h, w),
                    dtype=np.float32,
                )
                continue

            observation_spaces[mapped_key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)
        self.action_space = gym.spaces.Box(
            low=-np.ones(len(self.action_keys), dtype=np.float32),
            high=np.ones(len(self.action_keys), dtype=np.float32),
            shape=(len(self.action_keys),),
            dtype=np.float32,
        )

    @staticmethod
    def _candidate_raw_keys(source_key: str, mapped_key: str) -> list[str]:
        candidates = [source_key, mapped_key]
        for key in (source_key, mapped_key):
            if key.startswith("observation."):
                candidates.append(key[len("observation.") :])
            if key.startswith(f"{OBS_IMAGES}."):
                candidates.append(key[len(f"{OBS_IMAGES}.") :])
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _resolve_state_tensor(
        self,
        raw_observation: dict[str, Any],
        source_key: str,
        mapped_key: str,
        shape: tuple[int, ...],
        names: list[str] | None,
    ) -> torch.Tensor:
        if names:
            values = [float(raw_observation[key]) for key in names]
            arr = np.asarray(values, dtype=np.float32)
        else:
            arr = None
            for key in self._candidate_raw_keys(source_key, mapped_key):
                raw_value = self._to_numpy(raw_observation.get(key))
                if raw_value is not None:
                    arr = raw_value.astype(np.float32).reshape(-1)
                    break
            if arr is None:
                breakpoint()
                raise ValueError(f"{source_key} is missing from env observation")

        expected_dim = int(np.prod(shape))
        arr = arr.reshape(-1)
        if arr.size < expected_dim:
            arr = np.pad(arr, (0, expected_dim - arr.size), mode="constant")
        elif arr.size > expected_dim:
            arr = arr[:expected_dim]

        return torch.from_numpy(arr.reshape(shape).astype(np.float32))

    def _resolve_visual_tensor(
        self,
        raw_observation: dict[str, Any],
        source_key: str,
        mapped_key: str,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        expected_shape = shape
        if shape[-1] in (1, 3):
            expected_shape = (shape[-1], shape[0], shape[1])

        image = None
        for key in self._candidate_raw_keys(source_key, mapped_key):
            image = self._to_numpy(raw_observation.get(key))
            if image is not None:
                break

        if image is None:
            raise ValueError(f"{source_key} is missing from env observation")

        if image.ndim != 3:
            raise ValueError(f"image.ndim {image.ndim} != 3")

        image_tensor = torch.from_numpy(image)
        if image_tensor.shape[-1] in (1, 3):
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()

        image_tensor = image_tensor.to(dtype=torch.float32)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        if tuple(image_tensor.shape) != expected_shape:
            # image_tensor = torch.nn.functional.interpolate(
            #     image_tensor.unsqueeze(0),
            #     size=(expected_shape[1], expected_shape[2]),
            #     mode="bilinear",
            #     align_corners=False,
            # ).squeeze(0)
            # if image_tensor.shape[0] != expected_shape[0]:
            #     if image_tensor.shape[0] > expected_shape[0]:
            #         image_tensor = image_tensor[: expected_shape[0]]
            #     else:
            #         pad = expected_shape[0] - image_tensor.shape[0]
            #         image_tensor = torch.cat(
            #             [image_tensor, torch.zeros((pad, *image_tensor.shape[1:]), dtype=torch.float32)],
            #             dim=0,
            #         )
            raise ValueError(f"{image_tensor.shape} != {expected_shape}")

        return image_tensor

    def _project_observation(self, raw_observation: dict[str, Any]) -> RobotObservation:
        projected: RobotObservation = {}
        for source_key, mapped_key, feature_type, shape, names in self._observation_specs:
            if feature_type == FeatureType.VISUAL:
                projected[mapped_key] = self._resolve_visual_tensor(
                    raw_observation=raw_observation,
                    source_key=source_key,
                    mapped_key=mapped_key,
                    shape=shape,
                )
            else:
                projected[mapped_key] = self._resolve_state_tensor(
                    raw_observation=raw_observation,
                    source_key=source_key,
                    mapped_key=mapped_key,
                    shape=shape,
                    names=names,
                )
        return projected

    def _get_observation(self) -> RobotObservation:
        raw_observation = self.robot.get_observation()
        state = [
            raw_observation.pop("tcp_pose.position.x"),
            raw_observation.pop("tcp_pose.position.y"),
            raw_observation.pop("tcp_pose.position.z"),
            raw_observation.pop("tcp_pose.orientation.x"),
            raw_observation.pop("tcp_pose.orientation.y"),
            raw_observation.pop("tcp_pose.orientation.z"),
            raw_observation.pop("tcp_pose.orientation.w"),
            raw_observation.pop("tcp_velocity.linear.x"),
            raw_observation.pop("tcp_velocity.linear.y"),
            raw_observation.pop("tcp_velocity.linear.z"),
            raw_observation.pop("tcp_velocity.angular.x"),
            raw_observation.pop("tcp_velocity.angular.y"),
            raw_observation.pop("tcp_velocity.angular.z"),
            raw_observation.pop("tcp_error.x"),
            raw_observation.pop("tcp_error.y"),
            raw_observation.pop("tcp_error.z"),
            raw_observation.pop("tcp_error.rx"),
            raw_observation.pop("tcp_error.ry"),
            raw_observation.pop("tcp_error.rz"),
            raw_observation.pop("joint_positions.0"),
            raw_observation.pop("joint_positions.1"),
            raw_observation.pop("joint_positions.2"),
            raw_observation.pop("joint_positions.3"),
            raw_observation.pop("joint_positions.4"),
            raw_observation.pop("joint_positions.5"),
            raw_observation.pop("joint_positions.6"),
        ]
        state = np.asarray(state)
        raw_observation["state"] = state
        
        if not isinstance(raw_observation, dict):
            raw_observation = {}
        return self._project_observation(raw_observation)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotObservation, dict[str, Any]]:
        # use speical `AICCheatCodeTeleop` method to reset env through gz ros and aic-bringup
        self.teleop_device.reset_gz_sim()

        if self.reset_time_s > 0:
            time.sleep(self.reset_time_s)

        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        if isinstance(action, dict):
            action_dict = {key: float(value) for key, value in action.items()}
        else:
            if isinstance(action, torch.Tensor):
                action_values = action.detach().cpu().reshape(-1).tolist()
            else:
                action_values = np.asarray(action, dtype=np.float32).reshape(-1).tolist()

            if len(action_values) != len(self.action_keys):
                raise ValueError(
                    f"AIC action length mismatch: expected {len(self.action_keys)}, got {len(action_values)}."
                )
            action_dict = {
                self.action_keys[idx]: float(action_values[idx]) for idx in range(len(self.action_keys))
            }

        self.robot.send_action(action_dict)
        obs = self._get_observation()
        self.current_step += 1
        return obs, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False}

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        return self._raw_joint_positions

