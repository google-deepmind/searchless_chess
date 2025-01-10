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

"""An example training script."""

from collections.abc import Sequence

from absl import app
from absl import flags

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import metrics_evaluator
from searchless_chess.src import tokenizer
from searchless_chess.src import training
from searchless_chess.src import transformer
from searchless_chess.src import utils


_POLICY = flags.DEFINE_enum(
    'policy',
    'action_value',
    config_lib.POLICY_TYPES,
    'The policy used to play moves with the model.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  policy: config_lib.PolicyType = _POLICY.value  # pytype: disable=annotation-type-mismatch
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
      num_heads=4,
      num_layers=4,
      embedding_dim=64,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )
  train_config = config_lib.TrainConfig(
      learning_rate=1e-4,
      data=config_lib.DataConfig(
          batch_size=256,
          shuffle=True,
          worker_count=0,  # 0 disables multiprocessing.
          num_return_buckets=num_return_buckets,
          policy=policy,
          split='train',
      ),
      log_frequency=1,
      num_steps=20,
      ckpt_frequency=5,
      save_frequency=10,
  )
  eval_config = config_lib.EvalConfig(
      data=config_lib.DataConfig(
          batch_size=1,
          shuffle=False,
          worker_count=0,  # 0 disables multiprocessing.
          num_return_buckets=num_return_buckets,
          policy=None,  # pytype: disable=wrong-arg-types
          split='test',
      ),
      use_ema_params=True,
      policy=policy,
      batch_size=32,
      num_return_buckets=num_return_buckets,
      num_eval_data=64,
  )

  params = training.train(
      train_config=train_config,
      predictor_config=predictor_config,
      build_data_loader=data_loader.build_data_loader,
  )

  predictor = transformer.build_transformer_predictor(predictor_config)
  evaluator = metrics_evaluator.build_evaluator(predictor, eval_config)
  print(evaluator.step(params=params, step=train_config.num_steps))


if __name__ == '__main__':
  app.run(main)
