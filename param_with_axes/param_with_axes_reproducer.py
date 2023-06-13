"""Tests for the scale layers with partitioning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from functools import partial
import re

import argparse
import optax
import os
import time

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
from jax import lax
from jax import random

import flax
from flax import core
from flax import linen as nn
from flax import struct
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
from flax.linen import partitioning as nn_partitioning
from flax.linen import spmd
from flax import traverse_util
from flax.core.frozen_dict import FrozenDict

from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils

from typing import Callable, Iterable, Optional, Dict, Union, Any, Tuple

import numpy as np

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

# Type annotations
Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
ActivationFn = Callable[..., Array]
DotGeneralT = Callable[..., Array]
Collection = Union[Dict, FrozenDict]

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--axes', action='store_true', help='param_with_axes')
args = parser.parse_args()

use_param_with_axes = args.axes
print("DEBUG: param_with_axes", use_param_with_axes)

def _validate_params_axes(params_axes, params):
  axis_names = nn_partitioning.get_axis_names(params_axes)
  missing_params_axes = (
      set(traverse_util.flatten_dict(params, sep='/')) -
      set(traverse_util.flatten_dict(axis_names, sep='/')))
  if missing_params_axes:
    raise ValueError(
        f'Missing axis names for parameters: {missing_params_axes}')

def _split_scale_and_others(params):
  flt_scale = {}
  flt_other = {}
  flt_params = traverse_util.flatten_dict(params, sep='/')
  for k, v in flt_params.items():
    if k.endswith('_scale_meta'):
      flt_scale[k] = v
    else:
      flt_other[k] = v
  scale_params = traverse_util.unflatten_dict(flt_scale, sep='/')
  other_params = traverse_util.unflatten_dict(flt_other, sep='/')
  return core.freeze(scale_params), core.freeze(other_params)

def _merge_scale_and_others(scale_params, others):
  flt_scale = traverse_util.flatten_dict(scale_params, sep='/')
  flt_other = traverse_util.flatten_dict(others, sep='/')
  flt_params = {**flt_scale, **flt_other}
  return traverse_util.unflatten_dict(flt_params, sep='/')

class TrainState(struct.PyTreeNode):
  """
  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    params: The params that will be updated by the `tx`.
    scale_params: The scale_meta params that will be replaced by their grads.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
    params_axes: Contains axis metadata (e.g., names) matching `params` tree.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  params_axes: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  scale_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

  def variables(self) -> core.FrozenDict[str, Any]:
    variables = {}
    variables['params'] = _merge_scale_and_others(self.scale_params, self.params)
    return core.freeze(variables)

  def apply_gradients(self, *, grads, **kwargs):
    scale_grads, other_grads = _split_scale_and_others(grads['params'])

    updates, new_opt_state = self.tx.update(
        other_grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)

    return self.replace(
        step=self.step + 1,
        params=new_params,
        scale_params=scale_grads,
        opt_state=new_opt_state,
    )

  @classmethod
  def create(cls, apply_fn, model_variables, tx):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    other_variables, params = core.pop(model_variables, 'params')
    scale_params, other_params = _split_scale_and_others(params)

    if 'params_axes' in other_variables:
      other_variables, params_axes = core.pop(
          other_variables, 'params_axes'
      )
      _validate_params_axes(params_axes, other_params)
    else:
      params_axes = None

    if len(other_variables) > 0:
      raise ValueError(f'Contains unknown variables: {other_variables.keys}')

    opt_state = tx.init(other_params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=other_params,
        scale_params=scale_params,
        tx=tx,
        opt_state=opt_state,
        params_axes=params_axes,
    )

def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])

def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def input_scaling(compute_dtype, inp, scale, amax_history):
  return inp * scale

def input_scaling_fwd(compute_dtype, inp, scale, amax_history):
  qin = inp * scale
  new_scale = scale * 1.001
  new_amax_history = amax_history
  new_amax_history.at[-1].set(qin[-1][-1])
  return qin, (new_scale, new_amax_history)

def input_scaling_bwd(compute_dtype, res, g):
  new_scale, new_amax_history = res
  q_g = g
  return q_g, new_scale, new_amax_history

input_scaling.defvjp(input_scaling_fwd, input_scaling_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def output_scaling(compute_dtype, out, scale, amax_history):
  return out

def output_scaling_fwd(compute_dtype, out, scale, amax_history):
  return out, (scale, amax_history)

def output_scaling_bwd(compute_dtype, res, g):
  scale, amax_history = res
  new_scale = scale * 1.001
  q_g = scale * g
  new_amax_history = amax_history
  new_amax_history.at[-1].set(q_g[-1][-1])
  return q_g, new_scale, new_amax_history

output_scaling.defvjp(output_scaling_fwd, output_scaling_bwd)

class DenseGeneral(nn.Module):
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = \
      nn.initializers.lecun_normal()
  dot_general: DotGeneralT = lax.dot_general
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    original_shape = inputs.shape
    assert len(original_shape) >= 2

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                          np.prod(features))

    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        self.param_dtype,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)

    scale_args = (
        nn.initializers.ones_init(),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        nn.initializers.zeros_init(),
        (4,),
        jnp.float32,
    )

    if use_param_with_axes:
      input_scale = param_with_axes(
          'input_scale', *scale_args, axes=())
      kernel_scale = param_with_axes(
          'kernel_scale', *scale_args, axes=())
      output_grad_scale = param_with_axes(
          'output_grad_scale', *scale_args, axes=())
      input_amax_history = param_with_axes(
          'input_amax_history', *amax_history_args, axes=('scale_params',))
      kernel_amax_history = param_with_axes(
          'kernel_amax_history',
          *amax_history_args, axes=('scale_params',))
      output_grad_amax_history = param_with_axes(
          'output_grad_amax_history',
          *amax_history_args, axes=('scale_params',))
    else: 
      input_scale = self.param('input_scale_scale_meta', *scale_args)
      kernel_scale = self.param('kernel_scale_scale_meta', *scale_args)
      output_grad_scale = self.param('output_grad_scale_scale_meta', *scale_args)
      input_amax_history = self.param(
          'input_amax_history_scale_meta', *amax_history_args)
      kernel_amax_history = self.param(
          'kernel_amax_history_scale_meta', *amax_history_args)
      output_grad_amax_history = self.param(
          'output_grad_amax_history_scale_meta', *amax_history_args)

    inputs, kernel = nn.dtypes.promote_dtype(inputs, kernel, dtype=self.dtype)

    inputs = input_scaling(self.dtype, inputs, input_scale, input_amax_history)
   
    kernel = input_scaling(self.dtype, kernel, kernel_scale, kernel_amax_history)
    # Actual dense layer math.
    out = lax.dot(inputs, kernel)

    return output_scaling(self.dtype, out, output_grad_scale, output_grad_amax_history)

rules = (('batch', 'data'),
         ('hidden', 'model'),
         ('scale_param', None),)

def run_me(iters):
  device_mesh = mesh_utils.create_device_mesh((1, 1))
  mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
  
  model = DenseGeneral(8192, use_bias=False, kernel_axes=('hidden', 'mlp'))
  
  x_data = random.normal(random.PRNGKey(0), (8192, 8192))
  dy_data = random.normal(random.PRNGKey(0), (8192, 8192))
  k = random.PRNGKey(0)
  
  spmd.set_logical_axis_rules(rules)
  
  initialized_variables = model.init(k, x_data)
  opt = optax.adam(learning_rate=.1)
  state = TrainState.create(model_variables=initialized_variables,tx=opt,apply_fn=None)
  
  def loss_fn(state, x, dy):
    x = spmd.with_logical_constraint(x, ('batch', 'embed'))
    dy = spmd.with_logical_constraint(dy, ('batch', 'mlp'))

    y = model.apply(state, x)
    loss = y * dy.astype(y.dtype)
    return jnp.sum(loss)
  
  pjit_step_fn = pjit(
      jax.value_and_grad(loss_fn, argnums=[0]),
  )

  with mesh:
    for _ in range(iters):
      loss, grads = pjit_step_fn(state.variables(), x_data, dy_data)
#      print(grads)
      state = state.apply_gradients(grads=grads[0])
  return grads[0]

warm_iters = 20
time_iters = 100
jax.block_until_ready(run_me(warm_iters))
# Timing runs
st = time.time()
jax.block_until_ready(run_me(time_iters))
elapsed_time = (time.time() - st) / time_iters * 1000
print(f"Mean time: {elapsed_time} ms")

