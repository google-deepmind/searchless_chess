# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines the configuration dataclasses."""

import dataclasses
from typing import Literal


PolicyType = Literal['action_value', 'state_value', 'behavioral_cloning']
POLICY_TYPES = ['action_value', 'state_value', 'behavioral_cloning']


@dataclasses.dataclass(kw_only=True)
class DataConfig:
  """Config for the data generation."""

  # The batch size for the sequences.
  batch_size: int
  # Whether to shuffle the dataset (shuffling is applied per epoch).
  shuffle: bool = False
  # The seed used for shuffling and transformations of the data.
  seed: int | None = 0
  # Whether to drop partial batches.
  drop_remainder: bool = False
  # The number of child processes launched to parallelize the transformations.
  worker_count: int | None = 0
  # The number of return buckets.
  num_return_buckets: int
  # The dataset split.
  split: Literal['train', 'test']
  # The policy used to create the dataset.
  policy: PolicyType
  # The number of records to read from the dataset (can be useful when, e.g.,
  # the dataset does not fit into memory).
  num_records: int | None = None


@dataclasses.dataclass(kw_only=True)
class TrainConfig:
  """Config for the training function."""

  # The data configuration for training.
  data: DataConfig
  # The learning rate for Adam.
  learning_rate: float
  # The gradient clipping value.
  max_grad_norm: float = 1.0
  # The number of gradient steps.
  num_steps: int
  # The frequency (in gradient steps) at which checkpoints should be saved
  # (`None` means there is no checkpointing).
  ckpt_frequency: int | None = None
  # If provided, the maximum number of checkpoints to keep.
  ckpt_max_to_keep: int | None = 1
  # The frequency (in gradient steps) at which checkpoints should be saved
  # permanently (`None` means all checkpoints are temporary).
  save_frequency: int | None = None
  # The frequency of logging in gradient steps (`None` means no logging).
  log_frequency: int | None = None


@dataclasses.dataclass(kw_only=True)
class EvalConfig:
  """Config for the evaluator."""

  # The data configuration for evaluation.
  data: DataConfig
  # How many data points to consider for evaluation.
  num_eval_data: int | None = None
  # Enables use of ema-ed params in eval.
  use_ema_params: bool = False
  # The policy used to play moves with the model.
  policy: PolicyType
  # The number of return buckets.
  num_return_buckets: int
  # The batch size for evaluation.
  batch_size: int | None = None
