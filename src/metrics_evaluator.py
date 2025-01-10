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

"""Evaluation for the Chess experiments, retrieving static metrics like losses."""

import abc
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import os
from typing import Any

from absl import logging
import chess
import haiku as hk
import numpy as np
import scipy.stats

from searchless_chess.src import bagz
from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src import utils
from searchless_chess.src.engines import engine
from searchless_chess.src.engines import neural_engines


@dataclasses.dataclass
class ChessStaticMetrics:
  """Metrics retrieved from a supervised chess data loader.

  In the following, s is the board, a are the legal actions, r is the
  (bucketized) win probability evaluated by Stockfish for the pair (s, a), and
  theta are the parameters of the agent. Note that we easily interexchange the
  term 'return' and 'win probability'.
  Also, win_prob(s, a) = E_bucket (P_theta(bucket | s, a))

  Attributes:
    fen: Example: "r1b1kbnr/pp3ppp/1qn1p3/1BppP3/3P4/5N2/PPP2PPP/RNBQK2R w KQkq
      - 4 6".
    action_accuracy: argmax Q(s, a) == a
    output_log_loss: * For action-value: - E_a (log_2 P_theta(r | s, a)) * For
      state-value: - log_2 P_theta(r | s) * For BC: - log_2 P_theta(a | s)
    kendall_tau:  Measuring the discrepancy between the order of actions (by
      Q-value) between the agent and the Stockfish reference. See
      https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient.
    entropy: - E_{r,a} (log_2 P_theta(r | s, a)). Similar to the log-loss, but
      we average over return buckets too.
    l2_win_prob_loss: For action-value and state-value only, E_a ((win_prob(s,
      a) - r)**2)
    consistency_loss: For state-value only, (win_prob(s) - max_{a}
      win_prob(s'))**2 where s' = next state from s taking action a.
  """

  fen: str
  action_accuracy: bool
  output_log_loss: float
  kendall_tau: float
  entropy: float

  # Only for the action-value and state-value models.
  l2_win_prob_loss: float | None
  # Only for the state-value model.
  consistency_loss: float | None


class ChessStaticMetricsEvaluator(constants.Evaluator, abc.ABC):
  """Evaluator to compute various metrics (see ChessStaticMetrics)."""

  def __init__(
      self,
      predictor: constants.Predictor,
      num_return_buckets: int,
      dataset_path: str,
      batch_size: int,
      num_eval_data: int | None = None,
  ) -> None:
    """Initializes the evaluator.

    Args:
      predictor: To be evaluated.
      num_return_buckets: The number of return buckets to use.
      dataset_path: The path of the dataset to use.
      batch_size: How many sequences to pass to the predictor at once.
      num_eval_data: If specified, the number of data points to use for
        evaluation. If unspecified, all data points are used.
    """
    self._predictor = predictor
    self._return_buckets_edges, self._return_buckets_values = (
        utils.get_uniform_buckets_edges_values(num_return_buckets)
    )
    self._dataset_path = dataset_path
    self._batch_size = batch_size

    # Initialized to None, until step is called for the first time.
    self._test_data: dict[str, Any] = self._retrieve_test_data()

    if num_eval_data is not None:
      if len(self._test_data) < num_eval_data:
        raise ValueError(
            f'Not enough evaluation data points: {len(self._test_data)} <'
            f' {num_eval_data}'
        )
      self._test_data = dict(list(self._test_data.items())[:num_eval_data])

  def _metrics_to_filtered_dict(
      self, metrics: Sequence[ChessStaticMetrics]
  ) -> Mapping[str, np.ndarray]:
    """Returns a dictionary of averaged relevant metrics."""
    metrics_dict = {}
    fields = [x.name for x in dataclasses.fields(ChessStaticMetrics)]
    filtered_fields = filter(lambda x: x != 'fen', fields)
    filtered_fields = filter(
        lambda x: getattr(metrics[0], x) is not None,
        filtered_fields,
    )
    for key in filtered_fields:
      metrics_dict[key] = np.mean([getattr(x, key) for x in metrics])
    return metrics_dict

  def step(self, params: hk.Params, step: int) -> Mapping[str, Any]:
    """Returns the results of evaluating the predictor with `params`."""
    logging.info('Step of the metrics evaluator.')
    self._predict_fn = neural_engines.wrap_predict_fn(
        predictor=self._predictor,
        params=params,
        batch_size=self._batch_size,
    )
    logging.info('Computing metrics for %i FENs...', len(self._test_data))
    all_metrics = [self._compute_metrics(fen) for fen in self._test_data]
    logging.info('All metrics computed. Writing evals dict.')

    metrics_dict = self._metrics_to_filtered_dict(all_metrics)
    return {'eval_' + key: value for key, value in metrics_dict.items()}

  @abc.abstractmethod
  def _retrieve_test_data(self) -> dict[str, Any]:
    """Retrieves and returns the test data."""

  @abc.abstractmethod
  def _compute_metrics(self, fen: str) -> ChessStaticMetrics:
    """Returns metrics for the agent."""


class ActionValueChessStaticMetricsEvaluator(ChessStaticMetricsEvaluator):
  """Evaluator for action value."""

  def _retrieve_test_data(
      self,
  ) -> dict[str, Sequence[np.ndarray]]:
    """Retrieves and returns the test data.

    Returns:
      - Boards represented as FEN strings.
      - A mapping from FEN strings (above) to a sequence of
      (legal action, stockfish score).
    """
    move_to_action = utils.MOVE_TO_ACTION
    bag_reader = bagz.BagReader(self._dataset_path)

    action_score_dict = collections.defaultdict(dict)
    for bytes_data in bag_reader:
      fen, move, win_prob = constants.CODERS['action_value'].decode(bytes_data)
      action = move_to_action[move]
      action_score_dict[fen][action] = win_prob

    to_remove = []
    for fen in action_score_dict:
      list_items = list(action_score_dict[fen].items())
      list_items.sort(key=lambda x: x[0])
      action_score_dict[fen] = np.array(list_items, dtype=np.float32)

      board = chess.Board(fen)
      legal_actions = action_score_dict[fen][:, 0].tolist()
      true_legal_moves = engine.get_ordered_legal_moves(board)
      true_legal_actions = [move_to_action[x.uci()] for x in true_legal_moves]
      # Check that the dataset contains the right legal actions.
      if true_legal_actions != legal_actions:
        to_remove.append(fen)

    fraction_removed = len(to_remove) / len(action_score_dict)
    for fen in to_remove:
      del action_score_dict[fen]
    logging.info('Removed %f of FENs, wrong legal actions.', fraction_removed)

    return dict(action_score_dict)

  def _compute_metrics(self, fen: str) -> ChessStaticMetrics:
    if not hasattr(self, '_predict_fn'):
      raise ValueError('Predictor is not initialized.')
    neural_engine = neural_engines.ActionValueEngine(
        self._return_buckets_values,
        self._predict_fn,
    )
    analysis_results = neural_engine.analyse(chess.Board(fen))
    return self._compute_metrics_from_analysis(analysis_results)

  def _compute_metrics_from_analysis(
      self,
      analysis_results: engine.AnalysisResult,
  ) -> ChessStaticMetrics:
    """Returns metrics for the action_value agent."""
    fen = analysis_results['fen']
    legal_actions_returns = self._test_data[fen][:, 1]

    return_buckets_log_probs = analysis_results['log_probs']
    return_buckets_probs = np.exp(return_buckets_log_probs)

    # Compute the entropy.
    mult_probs = return_buckets_probs * return_buckets_log_probs
    nat_batch_entropies = np.sum(-mult_probs, axis=-1)
    entropy = np.mean(nat_batch_entropies) / np.log(2)

    # Compute the L2 loss on centipawn scores.
    win_probs = np.inner(
        self._return_buckets_values,
        return_buckets_probs,
    )
    l2_return_loss = np.mean((win_probs - legal_actions_returns) ** 2)

    # Compute the return log loss.
    return_buckets = utils.compute_return_buckets_from_returns(
        returns=legal_actions_returns,
        bins_edges=self._return_buckets_edges,
    )
    true_return_log_probs = np.take_along_axis(
        return_buckets_log_probs, return_buckets[..., None], axis=-1
    )[..., 0]
    return_log_loss = np.mean(-true_return_log_probs / np.log(2), axis=0)

    # Compute the action accuracy.
    best_legal_actions = legal_actions_returns == np.max(legal_actions_returns)
    action_accuracy = best_legal_actions[np.argmax(win_probs)]

    # Compute the Kendall-Taus.
    if len(win_probs) == 1:
      # There is perfect agreement if only one action is available.
      kendall_tau = 1.0
    else:
      kendall_tau, _ = scipy.stats.kendalltau(
          x=np.argsort(win_probs),
          y=np.argsort(legal_actions_returns),
      )

    return ChessStaticMetrics(
        fen=fen,
        action_accuracy=action_accuracy,
        output_log_loss=return_log_loss,
        l2_win_prob_loss=l2_return_loss,
        consistency_loss=None,
        kendall_tau=kendall_tau,
        entropy=entropy,
    )


class StateValueChessStaticMetricsEvaluator(
    ActionValueChessStaticMetricsEvaluator
):
  """Evaluator for state value."""

  def _compute_metrics(self, fen: str) -> ChessStaticMetrics:
    """Returns metrics for the action_value agent."""
    if not hasattr(self, '_predict_fn'):
      raise ValueError('Predictor is not initialized.')
    neural_engine = neural_engines.StateValueEngine(
        self._return_buckets_values,
        self._predict_fn,
    )
    analysis_results = neural_engine.analyse(chess.Board(fen))
    metrics = super()._compute_metrics_from_analysis({
        'log_probs': analysis_results['next_log_probs'],
        'fen': analysis_results['fen'],
    })
    current_win_prob = np.inner(
        self._return_buckets_values,
        np.exp(analysis_results['current_log_probs']),
    )
    next_win_probs = np.inner(
        self._return_buckets_values,
        np.exp(analysis_results['next_log_probs']),
    )
    next_win_prob = np.max(next_win_probs)
    metrics.consistency_loss = (current_win_prob - next_win_prob) ** 2
    return metrics


class BCChessStaticMetricsEvaluator(ActionValueChessStaticMetricsEvaluator):
  """Evaluator for behavioral cloning."""

  def _compute_metrics(self, fen: str) -> ChessStaticMetrics:
    """Returns metrics for the action_value agent."""
    if not hasattr(self, '_predict_fn'):
      raise ValueError('Predictor is not initialized.')
    legal_actions_returns = self._test_data[fen][:, 1]
    best_action_index = np.argmax(legal_actions_returns)

    neural_engine = neural_engines.BCEngine(predict_fn=self._predict_fn)
    analysis_results = neural_engine.analyse(chess.Board(fen))
    action_log_probs = analysis_results['log_probs']
    action_probs = np.exp(action_log_probs)

    # Compute the entropy.
    entropy = -np.sum(action_probs * action_log_probs) / np.log(2)

    # Compute the action log loss.
    action_loss = -action_log_probs[best_action_index] / np.log(2)

    # Compute the action accuracy.
    best_legal_actions = legal_actions_returns == np.max(legal_actions_returns)
    action_accuracy = best_legal_actions[np.argmax(action_probs)]

    # Compute the Kendall-Taus.
    if len(action_probs) == 1:
      # There is perfect agreement if only one action is available.
      kendall_tau = 1.0
    else:
      kendall_tau, _ = scipy.stats.kendalltau(
          x=np.argsort(action_probs),
          y=np.argsort(legal_actions_returns),
      )

    return ChessStaticMetrics(
        fen=fen,
        action_accuracy=action_accuracy,
        output_log_loss=action_loss,
        l2_win_prob_loss=None,
        consistency_loss=None,
        kendall_tau=kendall_tau,
        entropy=entropy,
    )


# Follows the base_constants.EvaluatorBuilder protocol.
def build_evaluator(
    predictor: constants.Predictor,
    config: config_lib.EvalConfig,
) -> ChessStaticMetricsEvaluator:
  """Returns an evaluator from an eval config."""
  evaluator_by_policy = {
      'action_value': ActionValueChessStaticMetricsEvaluator,
      'state_value': StateValueChessStaticMetricsEvaluator,
      'behavioral_cloning': BCChessStaticMetricsEvaluator,
  }
  return evaluator_by_policy[config.policy](
      predictor=predictor,
      # We always use the action-value data for evaluation since it provides the
      # required information for all the metrics.
      dataset_path=os.path.join(
          os.getcwd(),
          f'../data/{config.data.split}/action_value_data.bag',
      ),
      num_return_buckets=config.num_return_buckets,
      num_eval_data=config.num_eval_data,
      batch_size=config.batch_size,
  )
