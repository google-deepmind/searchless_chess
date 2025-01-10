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

"""Implements a PyGrain DataLoader for chess data."""

import abc
import os

import grain.python as pygrain
import jax
import numpy as np

from searchless_chess.src import bagz
from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src import utils


def _process_fen(fen: str) -> np.ndarray:
  return tokenizer.tokenize(fen).astype(np.int32)


def _process_move(move: str) -> np.ndarray:
  return np.asarray([utils.MOVE_TO_ACTION[move]], dtype=np.int32)


def _process_win_prob(
    win_prob: float,
    return_buckets_edges: np.ndarray,
) -> np.ndarray:
  return utils.compute_return_buckets_from_returns(
      returns=np.asarray([win_prob]),
      bins_edges=return_buckets_edges,
  )


class ConvertToSequence(pygrain.MapTransform, abc.ABC):
  """Base class for converting chess data to a sequence of integers."""

  def __init__(self, num_return_buckets: int) -> None:
    super().__init__()
    self._return_buckets_edges, _ = utils.get_uniform_buckets_edges_values(
        num_return_buckets,
    )
    # The loss mask ensures that we only train on the return bucket.
    self._loss_mask = np.full(
        shape=(self._sequence_length,),
        fill_value=True,
        dtype=bool,
    )
    self._loss_mask[-1] = False

  @property
  @abc.abstractmethod
  def _sequence_length(self) -> int:
    raise NotImplementedError()


class ConvertBehavioralCloningDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  @property
  def _sequence_length(self) -> int:
    return tokenizer.SEQUENCE_LENGTH + 1  # (s) + (a)

  def map(
      self, element: bytes
  ) -> tuple[constants.Sequences, constants.LossMask]:
    fen, move = constants.CODERS['behavioral_cloning'].decode(element)
    state = _process_fen(fen)
    action = _process_move(move)
    sequence = np.concatenate([state, action])
    return sequence, self._loss_mask


class ConvertStateValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  @property
  def _sequence_length(self) -> int:
    return tokenizer.SEQUENCE_LENGTH + 1  # (s) +  (r)

  def map(
      self, element: bytes
  ) -> tuple[constants.Sequences, constants.LossMask]:
    fen, win_prob = constants.CODERS['state_value'].decode(element)
    state = _process_fen(fen)
    return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
    sequence = np.concatenate([state, return_bucket])
    return sequence, self._loss_mask


class ConvertActionValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  @property
  def _sequence_length(self) -> int:
    return tokenizer.SEQUENCE_LENGTH + 2  # (s) + (a) + (r)

  def map(
      self, element: bytes
  ) -> tuple[constants.Sequences, constants.LossMask]:
    fen, move, win_prob = constants.CODERS['action_value'].decode(element)
    state = _process_fen(fen)
    action = _process_move(move)
    return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
    sequence = np.concatenate([state, action, return_bucket])
    return sequence, self._loss_mask


_TRANSFORMATION_BY_POLICY = {
    'behavioral_cloning': ConvertBehavioralCloningDataToSequence,
    'action_value': ConvertActionValueDataToSequence,
    'state_value': ConvertStateValueDataToSequence,
}


# Follows the base_constants.DataLoaderBuilder protocol.
def build_data_loader(config: config_lib.DataConfig) -> pygrain.DataLoader:
  """Returns a data loader for chess from the config."""
  data_source = bagz.BagDataSource(
      os.path.join(
          os.getcwd(),
          f'../data/{config.split}/{config.policy}_data.bag',
      ),
  )

  if config.num_records is not None:
    num_records = config.num_records
    if len(data_source) < num_records:
      raise ValueError(
          f'[Process {jax.process_index()}]: The number of records requested'
          f' ({num_records}) is larger than the dataset ({len(data_source)}).'
      )
  else:
    num_records = len(data_source)

  sampler = pygrain.IndexSampler(
      num_records=num_records,
      shard_options=pygrain.NoSharding(),
      shuffle=config.shuffle,
      num_epochs=None,
      seed=config.seed,
  )
  transformations = (
      _TRANSFORMATION_BY_POLICY[config.policy](
          num_return_buckets=config.num_return_buckets
      ),
      pygrain.Batch(config.batch_size, drop_remainder=True),
  )
  return pygrain.DataLoader(
      data_source=data_source,
      sampler=sampler,
      operations=transformations,
      worker_count=config.worker_count,
      read_options=None,
  )
