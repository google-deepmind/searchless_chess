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

"""Implements some utility functions."""

import math

import chess
import numpy as np


# The lists of the strings of the row and columns of a chess board,
# traditionally named rank and file.
_CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def _compute_all_possible_actions() -> tuple[dict[str, int], dict[int, str]]:
  """Returns two dicts converting moves to actions and actions to moves.

  These dicts contain all possible chess moves.
  """
  all_moves = []

  # First, deal with the normal moves.
  # Note that this includes castling, as it is just a rook or king move from one
  # square to another.
  board = chess.BaseBoard.empty()
  for square in range(64):
    next_squares = []

    # Place the queen and see where it attacks (we don't need to cover the case
    # for a bishop, rook, or pawn because the queen's moves includes all their
    # squares).
    board.set_piece_at(square, chess.Piece.from_symbol('Q'))
    next_squares += board.attacks(square)

    # Place knight and see where it attacks
    board.set_piece_at(square, chess.Piece.from_symbol('N'))
    next_squares += board.attacks(square)
    board.remove_piece_at(square)

    for next_square in next_squares:
      all_moves.append(
          chess.square_name(square) + chess.square_name(next_square)
      )

  # Then deal with promotions.
  # Only look at the last ranks.
  promotion_moves = []
  for rank, next_rank in [('2', '1'), ('7', '8')]:
    for index_file, file in enumerate(_CHESS_FILE):
      # Normal promotions.
      move = f'{file}{rank}{file}{next_rank}'
      promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

      # Capture promotions.
      # Left side.
      if file > 'a':
        next_file = _CHESS_FILE[index_file - 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
      # Right side.
      if file < 'h':
        next_file = _CHESS_FILE[index_file + 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
  all_moves += promotion_moves

  move_to_action, action_to_move = {}, {}
  for action, move in enumerate(all_moves):
    assert move not in move_to_action
    move_to_action[move] = action
    action_to_move[action] = move

  return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_actions()
NUM_ACTIONS = len(MOVE_TO_ACTION)


def centipawns_to_win_probability(centipawns: int) -> float:
  """Returns the win probability (in [0, 1]) converted from the centipawn score.

  Reference: https://lichess.org/page/accuracy
  Well-known transformation, backed by real-world data.

  Args:
    centipawns: The chess score in centipawns.
  """
  return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)


def get_uniform_buckets_edges_values(
    num_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns edges and values of uniformly sampled buckets in [0, 1].

  Example: for num_buckets=4, it returns:
  edges=[0.25, 0.50, 0.75]
  values=[0.125, 0.375, 0.625, 0.875]

  Args:
    num_buckets: Number of buckets to create.
  """
  full_linspace = np.linspace(0.0, 1.0, num_buckets + 1)
  edges = full_linspace[1:-1]
  values = (full_linspace[:-1] + full_linspace[1:]) / 2
  return edges, values


def compute_return_buckets_from_returns(
    returns: np.ndarray,
    bins_edges: np.ndarray,
) -> np.ndarray:
  """Arranges the discounted returns into bins.

  The returns are put into the bins specified by `bin_edges`. The length of
  `bin_edges` is equal to the number of buckets minus 1. In case of a tie (if
  the return is exactly equal to an edge), we take the bucket right before the
  edge. See example below.
  This function is purely using np.searchsorted, so it's a good reference to
  look at.

  Examples:
  * bin_edges=[0.5] and returns=[0., 1.] gives the buckets [0, 1].
  * bin_edges=[-30., 30.] and returns=[-200., -30., 0., 1.] gives the buckets
    [0, 0, 1, 1].

  Args:
    returns: An array of discounted returns, rank 1.
    bins_edges: The boundary values of the return buckets, rank 1.

  Returns:
    An array of buckets, described as integers, rank 1.

  Raises:
    ValueError if `returns` or `bins_edges` are not of rank 1.
  """
  if len(returns.shape) != 1:
    raise ValueError(
        'The passed returns should be of rank 1. Got'
        f' rank={len(returns.shape)}.'
    )
  if len(bins_edges.shape) != 1:
    raise ValueError(
        'The passed bins_edges should be of rank 1. Got'
        f' rank{len(bins_edges.shape)}.'
    )
  return np.searchsorted(bins_edges, returns, side='left')
