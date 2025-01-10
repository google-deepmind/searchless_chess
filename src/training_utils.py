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

"""Utilities for the main training function, in train.py."""

import functools
import pathlib
from typing import Any

import chex
from grain import python as pygrain
import haiku as hk
import jax
from jax import numpy as jnp
import optax
import orbax.checkpoint as ocp

from searchless_chess.src import constants


def replicate(
    array_tree: chex.ArrayTree,
    sharding: jax.sharding.PositionalSharding,
) -> chex.ArrayDeviceTree:
  """Replicates the `array_tree` across all devices specified by `sharding`.

  In a multi-controller setting, we cannot simply use `jax.device_put` to
  replicate the `array_tree` since not all devices are addressable from every
  process.

  Args:
    array_tree: The `array_tree` to be replicated across devices.
    sharding: Describes how the array should be laid out across devices. Here,
      we just use it to specify that the `array_tree` should be replicated (not
      sharded).

  Returns:
    The distributed `array_tree`, replicated across all the devices.
  """
  return jax.tree.map(
      lambda array: jax.make_array_from_callback(
          array.shape, sharding.replicate(), lambda _: array
      ),
      array_tree,
  )


def make_loss_fn(predictor: constants.Predictor) -> Any:
  """Returns the loss function for `update_parameters`.

  Args:
    predictor: The predictor to evaluate.
  """

  def loss_fn(
      params: hk.Params,
      sequences: constants.Sequences,
      mask: constants.LossMask,
  ) -> jnp.float32:
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      sequences: The input of sequences to evaluate. See neural_predictors.py.
      mask: Mask to apply to the losses. True means the loss will not be
        computed there.
    """
    conditionals = predictor.predict(params=params, targets=sequences, rng=None)
    true_conditionals = jnp.take_along_axis(
        conditionals, sequences[..., None], axis=-1
    )[..., 0]
    true_conditionals = jnp.where(mask, 0.0, true_conditionals)
    marginals = jnp.sum(true_conditionals, axis=1)
    # We need to clip to avoid a division by 0 below.
    seq_lengths = jnp.clip(jnp.sum(1 - mask, axis=1), a_min=1)
    return -jnp.mean(marginals / seq_lengths)

  return loss_fn


def _update_ema(
    ema_value: jnp.float32,
    current_param_value: jnp.float32,
    ema_decay: float = 0.99,
) -> jnp.float32:
  # The below implementation is a more numerically stable version of:
  # ema_value = ema_decay * ema_value + (1 - ema_decay) * current_param_value.
  return ema_value - (1.0 - ema_decay) * (ema_value - current_param_value)


@functools.partial(jax.jit, static_argnames=('grad_fn', 'optimizer'))
def update_parameters(
    params: hk.Params,
    params_ema: hk.Params,
    opt_state: optax.OptState,
    sequences: constants.Sequences,
    loss_mask: constants.LossMask,
    grad_fn: Any,
    optimizer: optax.GradientTransformation,
) -> tuple[hk.Params, hk.Params, optax.OptState, jnp.float32, jnp.float32]:
  """Computes gradients and updates the parameters using the optimizer.

  Backpropagation is done on the whole sequence. The whole function is jitted.

  Args:
    params: The parameters of the model.
    params_ema: EMA of params used in evals.
    opt_state: The state of the optimizer.
    sequences: The sequences to evaluate.
    loss_mask: Mask to apply to the loss.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the
      predictor. It returns the gradient of the parameters for this batch of
      data, and extra values.
    optimizer: An optimizer that computes parameter updates from the gradients.

  Returns:
    The updated parameters, ema of parameters, optimizer state, the loss and
    the gradient norm.
  """
  loss, grad = grad_fn(params, sequences, loss_mask)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)
  grad_norm_unclipped = optax.global_norm(grad)

  new_params_ema = jax.tree.map(_update_ema, params_ema, new_params)
  return new_params, new_params_ema, new_opt_state, loss, grad_norm_unclipped


def get_checkpoint_manager(
    ckpt_frequency: int,
    max_to_keep: int | None = None,
    save_frequency: int | None = None,
    checkpoint_dir: str | None = None,
) -> ocp.CheckpointManager:
  """Returns a `CheckpointManager`, which can save and restore checkpoints.

  Args:
    ckpt_frequency: The frequency at which checkpoints should be saved.
    max_to_keep: If provided, the maximum number of checkpoints to keep.
    save_frequency: The frequency at which checkpoints should be persisted.
    checkpoint_dir: The directory to save checkpoints to. If `None`, the default
      directory is retrieved.

  Raises:
    ValueError if save_frequency is set to a non-None value, and is not a
    multiple of ckpt_frequency.
  """
  if checkpoint_dir is None:
    checkpoint_dir = '/tmp/checkpoints'

  if save_frequency is not None and save_frequency % ckpt_frequency != 0:
    raise ValueError(
        '`save_frequency` must be a multiple of `ckpt_frequency`. Got'
        f' save_frequency={save_frequency} and ckpt_frequency={ckpt_frequency}.'
    )
  options = ocp.CheckpointManagerOptions(
      save_interval_steps=ckpt_frequency,
      max_to_keep=max_to_keep,
      keep_period=save_frequency,
  )
  checkpointers = dict(
      params=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()),
      params_ema=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()),
      opt_state=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()),
      data_iter=ocp.Checkpointer(pygrain.PyGrainCheckpointHandler()),  # pytype: disable=wrong-arg-types
  )
  return ocp.CheckpointManager(
      directory=checkpoint_dir,
      checkpointers=checkpointers,
      options=options,
  )


def restore_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    params: hk.Params,
    params_ema: hk.Params,
    opt_state: optax.OptState,
    data_iter: pygrain.PyGrainDatasetIterator,
    sharding: jax.sharding.PositionalSharding,
) -> tuple[
    hk.Params, hk.Params, optax.OptState, pygrain.PyGrainDatasetIterator
]:
  """Returns the restored params and optimizer state from a checkpoint."""

  def make_abstract(array_tree: chex.ArrayTree) -> jax.ShapeDtypeStruct:
    abstract_array_tree = jax.eval_shape(lambda x: x, array_tree)
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(
            shape=x.shape,
            dtype=x.dtype,
            sharding=sharding.replicate(),
        ),
        abstract_array_tree,
    )

  restored = checkpoint_manager.restore(
      step=step,
      items=dict(
          params=make_abstract(params),
          params_ema=make_abstract(params_ema),
          opt_state=make_abstract(opt_state),
          data_iter=data_iter,
      ),
  )
  return (
      restored['params'],
      restored['params_ema'],
      restored['opt_state'],
      restored['data_iter'],
  )


def load_parameters(
    params: hk.Params,
    step: int = -1,
    use_ema_params: bool = False,
    checkpoint_dir: str | None = None,
) -> hk.Params:
  """Loads and returns parameters from CNS.

  Args:
    params: The parameters of the model.
    step: The step at which that checkpoint was saved. If -1, loads the largest
      available step.
    use_ema_params: Enables loading of ema-ed params
    checkpoint_dir: The directory to load parameters from. If `None`, the
      default directory is retrieved.

  Returns:
    The parameters saved at `step`.

  Raises:
    FileNotFoundError: If `step` is not saved for that run.
  """
  if checkpoint_dir is None:
    checkpoint_dir = '/tmp/checkpoints'

  # Set the step to the largest available step if required.
  checkpoint_steps = ocp.utils.checkpoint_steps(checkpoint_dir)
  if step == -1:
    step = checkpoint_steps[-1]
  elif step not in checkpoint_steps:
    raise FileNotFoundError(f'Checkpoint {step} not found in {checkpoint_dir}.')

  # Construct the restore_args to inform orbax about the desired sharding.
  restore_args = ocp.checkpoint_utils.construct_restore_args(params)

  # Restore the checkpoint and return the parameters.
  dir_name = 'params_ema' if use_ema_params else 'params'
  checkpoint_path = pathlib.Path(checkpoint_dir) / str(step) / dir_name

  checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
  return checkpointer.restore(checkpoint_path, restore_args=restore_args)
