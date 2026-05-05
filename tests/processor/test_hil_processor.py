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

import numpy as np
import torch

from lerobot.processor import AICInterventionActionProcessorStep, InterventionActionProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import TransitionKey


def test_aic_intervention_action_processor_maps_dict_with_configured_key_order():
    step = AICInterventionActionProcessorStep(
        action_keys=["position.x", "position.y", "position.z"],
        terminate_on_success=True,
    )
    transition = create_transition(
        action=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        info={
            TeleopEvents.IS_INTERVENTION: True,
            TeleopEvents.SUCCESS: True,
            TeleopEvents.RERECORD_EPISODE: True,
        },
        complementary_data={
            "teleop_action": {
                "position.x": 1.2,
                "position.y": -0.3,
                "position.z": 0.5,
            }
        },
    )

    processed = step(transition)

    torch.testing.assert_close(
        processed[TransitionKey.ACTION],
        torch.tensor([1.2, -0.3, 0.5], dtype=torch.float32),
    )
    assert processed[TransitionKey.DONE] is True
    assert processed[TransitionKey.REWARD] == 1.0
    assert processed[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is True
    assert processed[TransitionKey.INFO][TeleopEvents.RERECORD_EPISODE] is True
    assert processed[TransitionKey.INFO][TeleopEvents.SUCCESS] is True
    torch.testing.assert_close(
        processed[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"],
        processed[TransitionKey.ACTION],
    )


def test_aic_intervention_action_processor_supports_numpy_teleop_action():
    step = AICInterventionActionProcessorStep(
        action_keys=["a", "b", "c"],
        terminate_on_success=False,
    )
    transition = create_transition(
        action=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        info={TeleopEvents.IS_INTERVENTION: True, TeleopEvents.SUCCESS: True},
        complementary_data={"teleop_action": np.array([3.0, 2.0, 1.0], dtype=np.float32)},
    )

    processed = step(transition)

    torch.testing.assert_close(
        processed[TransitionKey.ACTION],
        torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32),
    )
    assert processed[TransitionKey.DONE] is False
    assert processed[TransitionKey.REWARD] == 1.0


def test_legacy_intervention_action_processor_behavior_is_preserved():
    step = InterventionActionProcessorStep(use_gripper=True, terminate_on_success=True)
    transition = create_transition(
        action=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={
            "teleop_action": {
                "delta_x": 0.4,
                "delta_y": -0.2,
                "delta_z": 0.1,
                "gripper": 1.0,
            }
        },
    )

    processed = step(transition)

    torch.testing.assert_close(
        processed[TransitionKey.ACTION],
        torch.tensor([0.4, -0.2, 0.1, 1.0], dtype=torch.float32),
    )
    assert processed[TransitionKey.DONE] is False
    assert processed[TransitionKey.REWARD] == 0.0
