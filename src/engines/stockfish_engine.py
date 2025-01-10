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

"""Implements a stockfish engine."""

import os

import chess

from searchless_chess.src.engines import engine


class StockfishEngine(engine.Engine):
  """The classical version of stockfish."""

  def __init__(
      self,
      limit: chess.engine.Limit,
  ) -> None:
    self._limit = limit
    self._skill_level = None
    bin_path = os.path.join(
        os.getcwd(),
        '../Stockfish/src/stockfish',
    )
    self._raw_engine = chess.engine.SimpleEngine.popen_uci(bin_path)

  def __del__(self) -> None:
    self._raw_engine.close()

  @property
  def limit(self) -> chess.engine.Limit:
    return self._limit

  @property
  def skill_level(self) -> int | None:
    return self._skill_level

  @skill_level.setter
  def skill_level(self, skill_level: int) -> None:
    self._skill_level = skill_level
    self._raw_engine.configure({'Skill Level': self._skill_level})

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns analysis results from stockfish."""
    return self._raw_engine.analyse(board, limit=self._limit)

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from stockfish."""
    best_move = self._raw_engine.play(board, limit=self._limit).move
    if best_move is None:
      raise ValueError('No best move found, something went wrong.')
    return best_move


class AllMovesStockfishEngine(StockfishEngine):
  """A version of stockfish that evaluates all moves individually."""

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns analysis results from stockfish."""
    scores = []
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    for move in sorted_legal_moves:
      results = self._raw_engine.analyse(
          board,
          limit=self._limit,
          root_moves=[move],
      )
      scores.append((move, results['score'].relative))
    return {'scores': scores}

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from stockfish."""
    scores = self.analyse(board)['scores']
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0]
