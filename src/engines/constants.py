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

"""Constants for the engines."""

import functools
import os

import chess
import chess.engine
import chess.pgn
from jax import random as jrandom
import numpy as np

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import lc0_engine
from searchless_chess.src.engines import neural_engines
from searchless_chess.src.engines import stockfish_engine


def _build_neural_engine(
    model_name: str,
    checkpoint_step: int = -1,
) -> neural_engines.NeuralEngine:
  """Returns a neural engine."""

  match model_name:
    case '9M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 256
      num_heads = 8
    case '136M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 1024
      num_heads = 8
    case '270M':
      policy = 'action_value'
      num_layers = 16
      embedding_dim = 1024
      num_heads = 8
    case 'local':
      policy = 'action_value'
      num_layers = 4
      embedding_dim = 64
      num_heads = 4
    case _:
      raise ValueError(f'Unknown model: {model_name}')

  num_return_buckets = 128

  match policy:
    case 'action_value':
      output_size = num_return_buckets
    case 'behavioral_cloning':
      output_size = utils.NUM_ACTIONS
    case 'state_value':
      output_size = num_return_buckets

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=output_size,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  predictor = transformer.build_transformer_predictor(config=predictor_config)
  checkpoint_dir = os.path.join(
      os.getcwd(),
      f'../checkpoints/{model_name}',
  )
  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=predictor.initial_params(
          rng=jrandom.PRNGKey(1),
          targets=np.ones((1, 1), dtype=np.uint32),
      ),
      step=checkpoint_step,
  )
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(
      num_return_buckets
  )
  return neural_engines.ENGINE_FROM_POLICY[policy](
      return_buckets_values=return_buckets_values,
      predict_fn=neural_engines.wrap_predict_fn(
          predictor=predictor,
          params=params,
          batch_size=1,
      ),
  )


ENGINE_BUILDERS = {
    'local': functools.partial(_build_neural_engine, model_name='local'),
    '9M': functools.partial(
        _build_neural_engine, model_name='9M', checkpoint_step=6_400_000
    ),
    '136M': functools.partial(
        _build_neural_engine, model_name='136M', checkpoint_step=6_400_000
    ),
    '270M': functools.partial(
        _build_neural_engine, model_name='270M', checkpoint_step=6_400_000
    ),
    'stockfish': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05)
    ),
    'stockfish_all_moves': lambda: stockfish_engine.AllMovesStockfishEngine(
        limit=chess.engine.Limit(time=0.05)
    ),
    'leela_chess_zero_depth_1': lambda: lc0_engine.AllMovesLc0Engine(
        limit=chess.engine.Limit(nodes=1),
    ),
    'leela_chess_zero_policy_net': lambda: lc0_engine.Lc0Engine(
        limit=chess.engine.Limit(nodes=1),
    ),
    'leela_chess_zero_400_sims': lambda: lc0_engine.Lc0Engine(
        limit=chess.engine.Limit(nodes=400),
    ),
}
