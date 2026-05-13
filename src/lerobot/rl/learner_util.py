# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import copy
import logging
import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey, create_transition
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ACTION,
    DONE,
    OBS_STATE,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import Transition, move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    init_logging,
)

from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService



# Utilities/Helpers functions


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations)

    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
    cfg: TrainRLServerPipelineConfig,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """

    if cfg.use_policy_pre_post_processors:
        policy_preprocessor, _ = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path else None,
            dataset_stats=getattr(cfg.policy, "dataset_stats", None),
        )
    else:
        policy_preprocessor = lambda x: x

    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # convert RobotAction to PolicyAction so it can be use for training
            if cfg.use_policy_pre_post_processors:
                # DataProcessorPipeline expects canonical batch format as input, where
                # observation entries are flattened top-level keys such as
                # "observation.*". Passing {"observation": {...}} makes
                # batch_to_transition() drop observations and raises in
                # ObservationProcessorStep.
                act_transition = {
                    **transition["state"],
                    ACTION: transition[ACTION],
                }
                act_transition = policy_preprocessor(act_transition)
                transition[ACTION] = act_transition[TransitionKey.ACTION]

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition)

            # Add to offline buffer if it's an intervention
            complementary_info = transition.get("complementary_info", {})
            is_intervention = complementary_info.get(
                TeleopEvents.IS_INTERVENTION,
                complementary_info.get(TeleopEvents.IS_INTERVENTION.value, False),
            )
            # if dataset_repo_id is not None and is_intervention:
            #     offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


def make_transition_processor_hook(
    cfg: TrainRLServerPipelineConfig,
) -> Callable[[Transition], Transition] | None:
    """Create an optional transition hook to align offline replay data with policy processors."""
    if not cfg.use_policy_pre_post_processors:
        return None
    
    policy_cfg = copy.deepcopy(cfg.policy)
    policy_cfg.device = 'cpu' # HACK: prepcoessr include a step to move data to same device same policy

    policy_preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path else None,
        dataset_stats=getattr(cfg.policy, "dataset_stats", None),
    )

    def _process_transition(transition: Transition) -> Transition:
        state_batch = dict(transition["state"])
        state_batch[ACTION] = transition[ACTION]

        processed_state_batch = policy_preprocessor(state_batch)
        processed_next_state = policy_preprocessor(dict(transition["next_state"]))

        # NOTE: applied norm and relative action convert to RobotAction send by actor.
        processed_action = processed_state_batch.pop(ACTION, transition[ACTION])
        if not isinstance(processed_action, torch.Tensor):
            raise ValueError("Expected tensor action after policy preprocessing in offline replay hook.")

        # HACK: policy_preprocessor will add fields like 'next.reward' into result for some reason.
        for k in list(processed_state_batch.keys()):
            if k not in state_batch:
                processed_state_batch.pop(k)
        for k in list(processed_next_state.keys()):
            if k not in state_batch:
                processed_next_state.pop(k)
        processed_next_state.pop(ACTION)
        
        processed_transition: Transition = {
            "state": processed_state_batch,
            ACTION: processed_action,
            "reward": transition["reward"],
            "next_state": processed_next_state,
            "done": transition["done"],
            "truncated": transition["truncated"],
            "complementary_info": transition.get("complementary_info"),
        }

        if processed_transition["done"] or processed_transition["truncated"]:
            policy_preprocessor.reset()

        return processed_transition

    return _process_transition


def make_post_processor_hook(
    cfg: TrainRLServerPipelineConfig,
) -> Callable[[dict], dict] | None:
    """Create an optional export hook that maps policy-space replay data back to robot space."""
    if not cfg.use_policy_pre_post_processors:
        return None

    policy_cfg = copy.deepcopy(cfg.policy)
    policy_cfg.device = "cpu"  # HACK: processors include a step to move data to policy device.

    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path else None,
        dataset_stats=getattr(cfg.policy, "dataset_stats", None),
    )
    # Keep only the postprocessor unnormalization step for state reverse mapping.
    state_postprocessor = copy.deepcopy(policy_postprocessor[:1])
    if state_postprocessor.steps and hasattr(state_postprocessor.steps[0], "features"):
        state_postprocessor.steps[0].features = {**cfg.policy.input_features, **cfg.policy.output_features}
    state_keys = tuple(cfg.policy.input_features.keys())

    def _process_frame(frame: dict) -> dict:
        processed_frame = dict(frame)

        state_batch = {key: processed_frame[key] for key in state_keys if key in processed_frame}
        if state_batch:
            # Reverse state normalization in policy space -> robot space.
            state_transition = state_postprocessor._forward(create_transition(observation=state_batch))
            reversed_state = state_transition[TransitionKey.OBSERVATION]
            for key in state_keys:
                if key in reversed_state:
                    processed_frame[key] = reversed_state[key]

            # Refresh metadata for stateful post-processing (e.g. relative->absolute actions).
            policy_preprocessor(dict(reversed_state))

        action = processed_frame.get(ACTION)
        if not isinstance(action, torch.Tensor):
            raise ValueError("Expected tensor action before policy postprocessing in replay export hook.")

        squeeze_action = action.ndim <= 1
        action_batch = action.unsqueeze(0) if squeeze_action else action
        processed_action = policy_postprocessor(action_batch)
        if not isinstance(processed_action, torch.Tensor):
            raise ValueError("Expected tensor action after policy postprocessing in replay export hook.")

        if squeeze_action and processed_action.ndim >= 2 and processed_action.shape[0] == 1:
            processed_action = processed_action.squeeze(0)
        processed_frame[ACTION] = processed_action

        done_value = processed_frame.get(DONE, False)
        if isinstance(done_value, torch.Tensor):
            done_value = bool(done_value.reshape(-1)[0].item())
        if done_value:
            policy_preprocessor.reset()
            policy_postprocessor.reset()

        return processed_frame

    return _process_frame


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info("Resume training load the online dataset")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # NOTE: In RL is possible to not have a dataset.
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    logging.info("Convert to a offline replay buffer")
    transition_processor_hook = make_transition_processor_hook(cfg=cfg)
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        transition_processor_hook=transition_processor_hook,
    )
    return offline_replay_buffer
