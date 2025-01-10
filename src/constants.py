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

"""Constants, interfaces, and types."""

import abc
from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, NamedTuple, Protocol

from apache_beam import coders
from grain import python as pygrain
import haiku as hk
import jaxtyping as jtp

from searchless_chess.src import config as config_lib


# Integer sequences of token ids.
Sequences = jtp.UInt32[jtp.Array, 'B T']

# The predictions are log-probabilities (natural logarithm) for the passed
# sequences. It can either be marginal log-probabilities (i.e. log P(s) for all
# sequences s in the batch), or full conditionals (i.e. log P(token | s_<t) for
# all sequence s, time t and token in the alphabet).
Marginals = jtp.Float32[jtp.Array, '*B']
Conditionals = jtp.Float32[jtp.Array, '*B T F']
Predictions = Marginals | Conditionals

# True means the loss will be masked there, i.e. we ignore it.
LossMask = jtp.Bool[jtp.Array, 'B T']


@dataclasses.dataclass
class Predictor:
  """Defines the predictor interface."""

  initial_params: Callable[..., hk.MutableParams]
  predict: Callable[..., Predictions]


class DataLoaderBuilder(Protocol):

  def __call__(self, config: config_lib.DataConfig) -> pygrain.DataLoader:
    """Returns a PyGrain data loader from the `config`."""


class Evaluator(abc.ABC):
  """Defines the interface of the evaluator that evaluates a predictor."""

  @abc.abstractmethod
  def step(self, params: hk.Params, step: int) -> Mapping[str, Any]:
    """Returns the results of evaluating the predictor with `params`."""


class EvaluatorBuilder(Protocol):

  def __call__(
      self,
      predictor: Predictor,
      config: config_lib.EvalConfig,
  ) -> Evaluator:
    """Returns an evaluator for the `predictor` and `config`.

    Args:
      predictor: The predictor to be evaluated. The training loop continuously
        saves the predictor's parameters, which are then loaded in the
        evaluation loop and passed to the evaluator's step method.
      config: The configuration of the evaluator.
    """


CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
CODERS['state_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
))
CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))
CODERS['behavioral_cloning'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
))


class BehavioralCloningData(NamedTuple):
  fen: str
  move: str


class StateValueData(NamedTuple):
  fen: str
  win_prob: float


class ActionValueData(NamedTuple):
  fen: str
  move: str
  win_prob: float
