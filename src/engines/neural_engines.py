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

"""Implements the neural engines, returning analysis metrics for input FENs."""

from collections.abc import Callable, Sequence

import chess
import haiku as hk
import jax
import jax.nn as jnn
import numpy as np
import scipy.special

from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine

# Input = tokenized FEN, Output = log-probs, depends on the agent.
PredictFn = Callable[[np.ndarray], np.ndarray]


class NeuralEngine(engine.Engine):
  """Base class for neural engines.

  Attributes:
    predict_fn: The function to get raw outputs from the model.
    temperature: For the softmax used to play moves.
  """

  def __init__(
      self,
      return_buckets_values: np.ndarray | None = None,
      predict_fn: PredictFn | None = None,
      temperature: float | None = None,
  ):
    self._return_buckets_values = return_buckets_values
    self.predict_fn = predict_fn
    self.temperature = temperature
    self._rng = np.random.default_rng()


def _update_scores_with_repetitions(
    board: chess.Board,
    scores: np.ndarray,
) -> None:
  """Updates the win-probabilities for a board given possible repetitions."""
  sorted_legal_moves = engine.get_ordered_legal_moves(board)
  for i, move in enumerate(sorted_legal_moves):
    board.push(move)
    # If the move results in a draw, associate 50% win prob to it.
    if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
      scores[i] = 0.5
    board.pop()


class ActionValueEngine(NeuralEngine):
  """Neural engine using a function P(r | s, a)."""

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns buckets log-probs for each action, and FEN."""
    # Tokenize the legal actions.
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    legal_actions = [utils.MOVE_TO_ACTION[x.uci()] for x in sorted_legal_moves]
    legal_actions = np.array(legal_actions, dtype=np.int32)
    legal_actions = np.expand_dims(legal_actions, axis=-1)
    # Tokenize the return buckets.
    dummy_return_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)
    # Tokenize the board.
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
    sequences = np.stack([tokenized_fen] * len(legal_actions))
    # Create the sequences.
    sequences = np.concatenate(
        [sequences, legal_actions, dummy_return_buckets],
        axis=1,
    )
    return {'log_probs': self.predict_fn(sequences)[:, -1], 'fen': board.fen()}

  def play(self, board: chess.Board) -> chess.Move:
    return_buckets_log_probs = self.analyse(board)['log_probs']
    return_buckets_probs = np.exp(return_buckets_log_probs)
    win_probs = np.inner(return_buckets_probs, self._return_buckets_values)
    _update_scores_with_repetitions(board, win_probs)
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    if self.temperature is not None:
      probs = scipy.special.softmax(win_probs / self.temperature, axis=-1)
      return self._rng.choice(sorted_legal_moves, p=probs)
    else:
      best_index = np.argmax(win_probs)
      return sorted_legal_moves[best_index]


class StateValueEngine(NeuralEngine):
  """Neural engine using a function P(r | s)."""

  def _get_value_log_probs(
      self,
      predict_fn: PredictFn,
      fens: Sequence[str],
  ) -> np.ndarray:
    tokenized_fens = list(map(tokenizer.tokenize, fens))
    tokenized_fens = np.stack(tokenized_fens, axis=0).astype(np.int32)
    dummy_return_buckets = np.zeros((len(fens), 1), dtype=np.int32)
    sequences = np.concatenate([tokenized_fens, dummy_return_buckets], axis=1)
    return predict_fn(sequences)[:, -1]

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Defines a policy that predicts action and action value."""
    current_value_log_probs = self._get_value_log_probs(
        self.predict_fn, [board.fen()]
    )[0]

    # We perform a search of depth 1 to get the Q-values.
    next_fens = []
    for move in engine.get_ordered_legal_moves(board):
      board.push(move)
      next_fens.append(board.fen())
      board.pop()
    next_values_log_probs = self._get_value_log_probs(
        self.predict_fn, next_fens
    )
    # Flip the probabilities of the return buckets as we want to compute -value.
    next_values_log_probs = np.flip(next_values_log_probs, axis=-1)

    return {
        'current_log_probs': current_value_log_probs,
        'next_log_probs': next_values_log_probs,
        'fen': board.fen(),
    }

  def play(self, board: chess.Board) -> chess.Move:
    next_log_probs = self.analyse(board)['next_log_probs']
    next_probs = np.exp(next_log_probs)
    win_probs = np.inner(next_probs, self._return_buckets_values)
    _update_scores_with_repetitions(board, win_probs)
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    if self.temperature is not None:
      probs = scipy.special.softmax(win_probs / self.temperature, axis=-1)
      return self._rng.choice(sorted_legal_moves, p=probs)
    else:
      best_index = np.argmax(win_probs)
      return sorted_legal_moves[best_index]


class BCEngine(NeuralEngine):
  """Defines a policy that predicts action probs."""

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Defines a policy that predicts action probs."""
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
    tokenized_fen = np.expand_dims(tokenized_fen, axis=0)
    dummy_actions = np.zeros((1, 1), dtype=np.int32)
    sequences = np.concatenate([tokenized_fen, dummy_actions], axis=1)
    total_action_log_probs = self.predict_fn(sequences)[0, -1]
    assert len(total_action_log_probs) == utils.NUM_ACTIONS

    # We must renormalize the output distribution to only the legal moves.
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    legal_actions = [utils.MOVE_TO_ACTION[x.uci()] for x in sorted_legal_moves]
    legal_actions = np.array(legal_actions, dtype=np.int32)
    action_log_probs = total_action_log_probs[legal_actions]
    action_log_probs = jnn.log_softmax(action_log_probs)
    assert len(action_log_probs) == len(list(board.legal_moves))
    return {'log_probs': action_log_probs, 'fen': board.fen()}

  def play(self, board: chess.Board) -> chess.Move:
    action_log_probs = self.analyse(board)['log_probs']
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    if self.temperature is not None:
      probs = scipy.special.softmax(
          action_log_probs / self.temperature, axis=-1
      )
      return self._rng.choice(sorted_legal_moves, p=probs)
    else:
      best_index = np.argmax(action_log_probs)
      return sorted_legal_moves[best_index]


def wrap_predict_fn(
    predictor: constants.Predictor,
    params: hk.Params,
    batch_size: int = 32,
) -> PredictFn:
  """Returns a simple prediction function from a predictor and parameters.

  Args:
    predictor: Used to predict outputs.
    params: Neural network parameters.
    batch_size: How many sequences to pass to the predictor at once.
  """
  jitted_predict_fn = jax.jit(predictor.predict)

  def fixed_predict_fn(sequences: np.ndarray) -> np.ndarray:
    """Wrapper around the predictor `predict` function."""
    assert sequences.shape[0] == batch_size
    return jitted_predict_fn(
        params=params,
        targets=sequences,
        rng=None,
    )

  def predict_fn(sequences: np.ndarray) -> np.ndarray:
    """Wrapper to collate batches of sequences of fixed size."""
    remainder = -len(sequences) % batch_size
    padded = np.pad(sequences, ((0, remainder), (0, 0)))
    sequences_split = np.split(padded, len(padded) // batch_size)
    all_outputs = []
    for sub_sequences in sequences_split:
      all_outputs.append(fixed_predict_fn(sub_sequences))
    outputs = np.concatenate(all_outputs, axis=0)
    assert len(outputs) == len(padded)
    return outputs[: len(sequences)]  # Crop the padded sequences.

  return predict_fn


ENGINE_FROM_POLICY = {
    'action_value': ActionValueEngine,
    'state_value': StateValueEngine,
    'behavioral_cloning': BCEngine,
}
