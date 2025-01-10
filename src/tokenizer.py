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

"""Implements tokenization of FEN strings."""

import jaxtyping as jtp
import numpy as np


# pyfmt: disable
_CHARACTERS = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'p',
    'n',
    'r',
    'k',
    'q',
    'P',
    'B',
    'N',
    'R',
    'Q',
    'K',
    'w',
    '.',
]
# pyfmt: enable
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 77


def tokenize(fen: str) -> jtp.Int32[jtp.Array, 'T']:
  """Returns an array of tokens from a fen string.

  We compute a tokenized representation of the board, from the FEN string.
  The final array of tokens is a mapping from this string to numbers, which
  are defined in the dictionary `_CHARACTERS_INDEX`.
  For the 'en passant' information, we convert the '-' (which means there is
  no en passant relevant square) to '..', to always have two characters, and
  a fixed length output.

  Args:
    fen: The board position in Forsyth-Edwards Notation.
  """
  # Extracting the relevant information from the FEN.
  board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
  board = board.replace('/', '')
  board = side + board

  indices = list()

  for char in board:
    if char in _SPACES_CHARACTERS:
      indices.extend(int(char) * [_CHARACTERS_INDEX['.']])
    else:
      indices.append(_CHARACTERS_INDEX[char])

  if castling == '-':
    indices.extend(4 * [_CHARACTERS_INDEX['.']])
  else:
    for char in castling:
      indices.append(_CHARACTERS_INDEX[char])
    # Padding castling to have exactly 4 characters.
    if len(castling) < 4:
      indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

  if en_passant == '-':
    indices.extend(2 * [_CHARACTERS_INDEX['.']])
  else:
    # En passant is a square like 'e3'.
    for char in en_passant:
      indices.append(_CHARACTERS_INDEX[char])

  # Three digits for halfmoves (since last capture) is enough since the game
  # ends at 50.
  halfmoves_last += '.' * (3 - len(halfmoves_last))
  indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

  # Three digits for full moves is enough (no game lasts longer than 999
  # moves).
  fullmoves += '.' * (3 - len(fullmoves))
  indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

  assert len(indices) == SEQUENCE_LENGTH

  return np.asarray(indices, dtype=np.uint8)
