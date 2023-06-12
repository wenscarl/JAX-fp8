"""Tests for the fp8 layers with partitioning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from functools import partial
import re

import argparse
import optax
import os

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

def _split_fp8_and_others(params):
  flt_fp8 = {}
  flt_other = {}
  flt_params = traverse_util.flatten_dict(params, sep='/')
  for k, v in flt_params.items():
    if k.endswith('_fp8_meta'):
      flt_fp8[k] = v
    else:
      flt_other[k] = v
  fp8_params = traverse_util.unflatten_dict(flt_fp8, sep='/')
  other_params = traverse_util.unflatten_dict(flt_other, sep='/')
  return core.freeze(fp8_params), core.freeze(other_params)

def _merge_fp8_and_others(fp8_params, others):
  flt_fp8 = traverse_util.flatten_dict(fp8_params, sep='/')
  flt_other = traverse_util.flatten_dict(others, sep='/')
  flt_params = {**flt_fp8, **flt_other}
  return traverse_util.unflatten_dict(flt_params, sep='/')

class TrainState(struct.PyTreeNode):
  """
  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    params: The params that will be updated by the `tx`.
    fp8_params: The fp8_meta params that will be replaced by their grads.
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
  fp8_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

  def variables(self) -> core.FrozenDict[str, Any]:
    variables = {}
    variables['params'] = _merge_fp8_and_others(self.fp8_params, self.params)
    return core.freeze(variables)

  def apply_gradients(self, *, grads, **kwargs):
    fp8_grads, other_grads = _split_fp8_and_others(grads['params'])

    updates, new_opt_state = self.tx.update(
        other_grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)

    return self.replace(
        step=self.step + 1,
        params=new_params,
        fp8_params=fp8_grads,
        opt_state=new_opt_state,
    )

  @classmethod
  def create(cls, apply_fn, model_variables, tx):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    other_variables, params = core.pop(model_variables, 'params')
    fp8_params, other_params = _split_fp8_and_others(params)

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
        fp8_params=fp8_params,
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

def get_fp8_max(fp8_dtype):
  assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  return jnp.finfo(fp8_dtype).max.astype(jnp.float32)

def quantize(x, q_dtype, scale, compute_dtype):
  # We need to explicitly cast the max value to compute_dtype, otherwise the jax
  # dtype promotion will cast the scaled_x to fp32 in the following ops, which
  # would violate the fp8-matmul pattern matching.
  dtype_max = get_fp8_max(q_dtype).astype(compute_dtype)

  scaled_x = x / scale.astype(compute_dtype)
  clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)

  return clipped_x.astype(q_dtype)

def dequantize(x, dq_dtype, scale):
  return x.astype(dq_dtype) * scale.astype(dq_dtype)

def quantize_dequantize(x, q_dtype, scale, compute_dtype):
  qx = quantize(x, q_dtype, scale, compute_dtype)
  return dequantize(qx, x.dtype, scale)

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  exp = jnp.floor(jnp.log2(fp8_max / amax)) - margin
  sf = jnp.round(lax.pow(2., jnp.abs(exp)))
  sf = jnp.where(amax > 0.0, sf, scale)
  sf = jnp.where(lax.is_finite(amax), sf, scale)
  sf = jnp.where(exp < 0, 1.0 / sf, sf)
  # The scaling factor we need equals to the notion of "scale_inv" in
  # TransformerEngine. So, we convert the sf to its reciprocal.
  return 1.0 / sf

def compute_scale_and_amax_history(x, q_dtype, scale, amax_history):
  dtype_max = get_fp8_max(q_dtype)

  amax_update = jnp.max(jnp.abs(x)).astype(scale.dtype)
  new_amax_history = \
      jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)

  amax_from_history = jnp.max(new_amax_history, axis=0)
  new_scale = compute_scale(amax_from_history, scale, dtype_max)
  return new_scale, new_amax_history

def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
  qx = quantize_dequantize(x, q_dtype, scale, compute_dtype)
  new_scale, new_amax_history = compute_scale_and_amax_history(
      x, q_dtype, scale, amax_history)
  return qx, new_scale, new_amax_history

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def in_qdq(compute_dtype, inp, scale, amax_history):
  qin, _, _ = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin

def in_qdq_fwd(compute_dtype, inp, scale, amax_history):
  qin, new_scale, new_amax_history = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin, (new_scale, new_amax_history)

def in_qdq_bwd(compute_dtype, res, g):
  new_scale, new_amax_history = res
  q_g = g
  return q_g, new_scale, new_amax_history

in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def out_qdq(compute_dtype, out, scale, amax_history):
  return out

def out_qdq_fwd(compute_dtype, out, scale, amax_history):
  return out, (scale, amax_history)

def out_qdq_bwd(compute_dtype, res, g):
  scale, amax_history = res
  q_g, new_scale, new_amax_history = qdq_and_return(
      g, jnp.float8_e5m2, scale, amax_history, compute_dtype)
  return q_g, new_scale, new_amax_history

out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)

class DenseGeneral(nn.Module):
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  amax_history_length: int = 16
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = \
      nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  activation: Optional[ActivationFn] = None
  dot_general: DotGeneralT = lax.dot_general
  kernel_axes: Tuple[str, ...] = ()
  bias_axes: Tuple[str, ...] = ()

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

    if self.use_bias:
      bias = param_with_axes(
          'bias',
          self.bias_init,
          (np.prod(features),),
          self.param_dtype,
          axes=self.bias_axes)
      bias = jnp.asarray(bias, self.dtype)
    else:
      bias = None

    scale_args = (
        nn.initializers.ones_init(),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        nn.initializers.zeros_init(),
        (self.amax_history_length,),
        jnp.float32,
    )

    if use_param_with_axes:
      input_amax_history = param_with_axes(
          'input_amax_history', *amax_history_args, axes=('fp8_params',))
      kernel_amax_history = param_with_axes(
          'kernel_amax_history',
          *amax_history_args, axes=('fp8_params',))
      output_grad_amax_history = param_with_axes(
          'output_grad_amax_history',
          *amax_history_args, axes=('fp8_params',))
  
      input_scale = param_with_axes(
          'input_scale', *scale_args, axes=())
      kernel_scale = param_with_axes(
          'kernel_scale', *scale_args, axes=())
      output_grad_scale = param_with_axes(
          'output_grad_scale', *scale_args, axes=())
    else: 
      input_amax_history = self.param(
          'input_amax_history_fp8_meta', *amax_history_args)
      kernel_amax_history = self.param(
          'kernel_amax_history_fp8_meta', *amax_history_args)
      output_grad_amax_history = self.param(
          'output_grad_amax_history_fp8_meta', *amax_history_args)
  
      input_scale = self.param('input_scale_fp8_meta', *scale_args)
      kernel_scale = self.param('kernel_scale_fp8_meta', *scale_args)
      output_grad_scale = self.param('output_grad_scale_fp8_meta', *scale_args)
  
      inputs, kernel, bias = nn.dtypes.promote_dtype(inputs, kernel, bias,
                                                     dtype=self.dtype)

    # Reshape the inputs to 2D matrix.
    inp_mat = jnp.reshape(inputs,
                          (-1, np.prod([inputs.shape[ax] for ax in axis])))
    inp_mat = in_qdq(self.dtype, inp_mat, input_scale,
                     input_amax_history)
    kernel = in_qdq(self.dtype, kernel, kernel_scale,
                    kernel_amax_history)

    # Actual dense layer math.
    out = lax.dot(inp_mat, kernel)

    out = out_qdq(self.dtype, out, output_grad_scale,
                  output_grad_amax_history)

    if self.use_bias:
      # The bias has already been promoted. So, if it is fp32, we need to cast
      # it to bf16 to trigger fp8 matmul fusion.
      if bias.dtype == jnp.float32:
        bias = bias.astype(jnp.bfloat16)
        bias = bias.astype(jnp.float32)
      out = out + bias

    if self.activation:
      out = self.activation(out)

    # Reshape back the outputs.
    out = jnp.reshape(out, (*original_shape[0:-len(axis)], *tuple(features)))

    return out

rules = (('batch', 'data'),
         ('hidden', 'model'),
         ('fp8_param', None),)

def run_me():
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
  iters = 500

  with mesh:
    for _ in range(iters):
      loss, grads = pjit_step_fn(state.variables(), x_data, dy_data)
#      print(grads)
      state = state.apply_gradients(grads=grads[0])
  return grads[0]

print(run_me())

