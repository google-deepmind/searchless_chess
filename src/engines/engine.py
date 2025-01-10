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

"""Defines the Engine interface."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import chess

from searchless_chess.src import utils

AnalysisResult = Mapping[str, Any]


def get_ordered_legal_moves(board: chess.Board) -> Sequence[chess.Move]:
  """Returns legal moves ordered by action value."""
  return sorted(board.legal_moves, key=lambda x: utils.MOVE_TO_ACTION[x.uci()])


class Engine(Protocol):

  def analyse(self, board: chess.Board) -> AnalysisResult:
    """Returns various analysis results (including output) from a model."""

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best legal move from a given board."""
