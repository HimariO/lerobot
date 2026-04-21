#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.types import EnvTransition, RobotObservation, TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature


class StrLookupTableStep(ObservationProcessorStep):
    """
    Turn `task` language instruction into single integer value that 
    can be use when we use embedding table as text encoder.
    ex: {"put apple into the box.": 0, "inerset cable into the port.": 1}
    """
    
    task_key: str = "task"
    TABLE = {
        "[AIC] plug cable into SFP_PORT_0": 0,
        "[AIC] plug cable into SFP_PORT_1": 1,
        "[AIC] plug cable into SC_PORT_0": 2,
        "[AIC] plug cable into SC_PORT_1": 3,
        "insert sfp_tip plug into nic_card_mount_1": 4,
        "TaskType.SFP_0": 5,
        "TaskType.SFP_1": 6,
        "TaskType.SC_0": 7,
        "TaskType.SC_1": 8,
    }

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the task description(s) from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of task strings, or None if the task key is not found or the value is None.
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")

        # Standardize to a list of strings for the tokenizer
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task

        return None

    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        """
        Detects the torch.device from existing tensors in the transition.

        It checks tensors in the observation dictionary first, then the action tensor.

        Args:
            transition: The environment transition.

        Returns:
            The detected `torch.device`, or None if no tensors are found.
        """
        # Check observation tensors first (most likely place to find tensors)
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device

        # Fallback to checking the action tensor
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device

        return None  # No tensors found, default will be CPU

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """
        A wrapper around the tokenizer call.

        Args:
            text: A string or list of strings to tokenize.

        Returns:
            A dictionary containing tokenized 'input_ids' and 'attention_mask' as PyTorch tensors.
        """
        if isinstance(text, list):
            input_ids = [[self.TABLE[s]] for s in text]
        else:
            input_ids = [[self.TABLE[text]]]
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        Tokenizes the task description and adds it to the observation dictionary.

        This method retrieves the task, tokenizes it, moves the resulting tensors to the
        same device as other data in the transition, and updates the observation.

        Args:
            observation: The original observation dictionary.

        Returns:
            The updated observation dictionary including token IDs and an attention mask.
        """
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # Tokenize the task (this will create CPU tensors)
        tokenized_prompt = self._tokenize_text(task)

        # Detect the device from existing tensors in the transition to ensure consistency
        target_device = self._detect_device(self.transition)

        # Move new tokenized tensors to the detected device
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }
        # Create a new observation dict to avoid modifying the original in place
        new_observation = dict(observation)

        # Add tokenized data to the observation
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds feature definitions for the language tokens and attention mask.

        This updates the policy features dictionary to include the new data added to the
        observation, ensuring downstream components are aware of their shape and type.

        Args:
            features: The dictionary of existing policy features.

        Returns:
            The updated dictionary of policy features.
        """
        # Add a feature for the token IDs if it doesn't already exist
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(1,)
            )

        # Add a feature for the attention mask if it doesn't already exist
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(1,)
            )

        return features


class FlattenMultiStepObs(ObservationProcessorStep):
    """
    Temporarily convert obervation field with shape [batch, observartion_steps, data_dim] 
     into 2D tensor with shape [batch * observartion_steps, data_dim]. So the remaining processor can handle 
     the data without compbilty issue.
    """
    
    n_obs_steps: int = 0

    def observation(self, observation: RobotObservation) -> RobotObservation:
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        pass



def make_multi_task_dit_pre_post_processors(
    config: MultiTaskDiTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a Multi-Task DiT policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Adding a batch dimension.
    3. Tokenizing the language task description (if present).
    4. Moving the data to the specified device.
    5. Normalizing the input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output features to their original scale.
    2. Moving the data to the CPU.

    Args:
        config: The configuration object for the Multi-Task DiT policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
        convert_relative_quat_to_rotvec=True,
    )
    if not config.text_embed_table:
        task_tokenize = TokenizerProcessorStep(
            tokenizer_name=config.text_encoder_name,
            padding=config.tokenizer_padding,
            padding_side=config.tokenizer_padding_side,
            max_length=config.tokenizer_max_length,
            truncation=config.tokenizer_truncation,
        )
    else:
        task_tokenize = StrLookupTableStep()

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        task_tokenize,
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
