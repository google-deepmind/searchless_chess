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

"""Launches a tournament between various engines to compute their Elos."""

from collections.abc import Mapping, Sequence
import copy
import datetime
import itertools
import os

from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import numpy as np

from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine
from searchless_chess.src.engines import stockfish_engine


_NUM_GAMES = flags.DEFINE_integer(
    name='num_games',
    default=None,
    help='The number of games to play between each pair of engines.',
    required=True,
)

# We use a stockfish engine to evaluate the current board and terminate the
# game early if the score is high enough (i.e., _MIN_SCORE_TO_STOP).
_EVAL_STOCKFISH_ENGINE = stockfish_engine.StockfishEngine(
    limit=chess.engine.Limit(time=0.01)
)
_MIN_SCORE_TO_STOP = 1300


def _play_game(
    engines: tuple[engine.Engine, engine.Engine],
    engines_names: tuple[str, str],
    white_name: str,
    initial_board: chess.Board | None = None,
) -> chess.pgn.Game:
  """Plays a game of chess between two engines.

  Args:
    engines: The engines to play the game.
    engines_names: The names of the engines.
    white_name: The name of the engine playing white.
    initial_board: The initial board (if None, the standard starting position).

  Returns:
    The game played between the engines.
  """
  if initial_board is None:
    initial_board = chess.Board()
  white_player = engines_names.index(white_name)
  current_player = white_player if initial_board.turn else 1 - white_player
  board = initial_board
  result = None
  print(f'Starting FEN: {board.fen()}')

  while not (
      board.is_game_over()
      or board.can_claim_fifty_moves()
      or board.is_repetition()
  ):
    best_move = engines[current_player].play(board)
    print(f'Best move: {best_move.uci()}')

    # Push move to the game.
    board.push(best_move)
    current_player = 1 - current_player

    # We analyse the board once the last move is done and pushed.
    info = _EVAL_STOCKFISH_ENGINE.analyse(board)
    score = info['score'].relative
    if score.is_mate():
      is_winning = score.mate() > 0
    else:
      is_winning = score.score() > 0
    score_too_high = score.is_mate() or abs(score.score()) > _MIN_SCORE_TO_STOP

    if score_too_high:
      is_white = board.turn == chess.WHITE
      if is_white and is_winning or (not is_white and not is_winning):
        result = '1-0'
      else:
        result = '0-1'
      break
  print(f'End FEN: {board.fen()}')

  game = chess.pgn.Game.from_board(board)
  game.headers['Event'] = 'UAIChess'
  game.headers['Date'] = datetime.datetime.today().strftime('%Y.%m.%d')
  game.headers['White'] = white_name
  game.headers['Black'] = engines_names[1 - white_player]
  if result is not None:  # Due to early stopping.
    game.headers['Result'] = result
  else:
    game.headers['Result'] = board.result(claim_draw=True)
  return game


def _run_tournament(
    engines: Mapping[str, engine.Engine],
    opening_boards: Sequence[chess.Board],
) -> Sequence[chess.pgn.Game]:
  """Runs a tournament between engines given openings.

  We play both sides for each opening, and the total number of games played per
  pair is therefore 2 * len(opening_boards).

  Args:
    engines: A mapping from engine names to engines.
    opening_boards: The boards to use as openings.

  Returns:
    The games played between all the engines.
  """
  games = list()

  for engine_name_0, engine_name_1 in itertools.combinations(engines, 2):
    print(f'Playing games between {engine_name_0} and {engine_name_1}')
    engine_0 = engines[engine_name_0]
    engine_1 = engines[engine_name_1]

    for opening_board, white_idx in itertools.product(opening_boards, (0, 1)):
      white_name = (engine_name_0, engine_name_1)[white_idx]
      game = _play_game(
          engines=(engine_0, engine_1),
          engines_names=(engine_name_0, engine_name_1),
          white_name=white_name,
          # Copy as we modify the opening board in the function.
          initial_board=copy.deepcopy(opening_board),
      )
      games.append(game)

  return games


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # To ensure variability in the games we play, we use the openings from the
  # Encyclopedia of Chess Openings.
  openings_path = os.path.join(
      os.getcwd(),
      '../data/eco_openings.pgn',
  )
  opening_boards = list()

  with open(openings_path, 'r') as file:
    while (game := chess.pgn.read_game(file)) is not None:
      opening_boards.append(game.end().board())

  # We subsample the openings according to the desired number of games.
  rng = np.random.default_rng(seed=1)
  opening_indices = rng.choice(
      np.arange(len(opening_boards)),
      # Divide by two as we consider both sides per opening (white and black).
      size=_NUM_GAMES.value // 2,
      replace=False,
  )
  opening_boards = list(opening_boards[idx] for idx in opening_indices)

  engines = {
      agent: constants.ENGINE_BUILDERS[agent]()
      for agent in [
          '9M',
          '136M',
          '270M',
          'stockfish',
          'stockfish_all_moves',
          'leela_chess_zero_depth_1',
          'leela_chess_zero_policy_net',
          'leela_chess_zero_400_sims',
      ]
  }

  games = _run_tournament(engines=engines, opening_boards=opening_boards)

  games_path = os.path.join(os.getcwd(), '../data/tournament_games.pgn')

  print(f'Writing games to {games_path}')
  with open(games_path, 'w') as file:
    for game in games:
      file.write(str(game))
      file.write('\n\n')


if __name__ == '__main__':
  app.run(main)
