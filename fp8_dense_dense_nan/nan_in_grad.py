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
from flax import linen as nn
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
from flax import traverse_util
from flax.core.frozen_dict import FrozenDict

from flax import struct, traverse_util, linen as nn
from typing import (Any, Callable, Iterable, List, Optional, Mapping, Sequence, Tuple, Union)
from typing import Callable, Iterable, Optional, Dict, Union, Any, Tuple
from flax import core

parser = argparse.ArgumentParser(description='Benchmark a basic encoder layer')
parser.add_argument('--fp8', action='store_true', help='Enable fp8')
parser.add_argument('--mixed', action='store_true',
                    help='Enable mixed precision and bf16 compute type')
args = parser.parse_args()

use_fp8 = args.fp8
use_mixed = args.mixed

# dtype is the base type when fp8 presents.
dtype = jnp.bfloat16 if use_mixed else jnp.float32

dim = 16

Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

FAKE_E4M3 = jnp.float8_e4m3fn
FAKE_E5M2 = jnp.float8_e5m2
E4M3_MAX = 448
E5M2_MAX = 57344


def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  elif fake_dtype == FAKE_E5M2:
    return E5M2_MAX
  else:
    raise ValueError('Only FAKE_E4M3 and FAKE_E5M2 supported')

def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  scaled_x = jnp.clip(x / scale,-dtype_max, dtype_max)
  return scaled_x.astype(quantized_dtype)

def dequantize(x, wide_dtype, scale):
  return x.astype(wide_dtype) * scale

def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def qdq_and_new_scale(x, dtype, scale):
  qx = quantize_dequantize(x, dtype, scale)
  new_scale = 1.1 / get_fp8_max(dtype)
  return qx, new_scale

@jax.custom_vjp
def input_qdq(kernel, kernel_scale):
  return kernel

def input_qdq_fwd(kernel, kernel_scale):
  qkernel, new_kernel_scale = qdq_and_new_scale(kernel, FAKE_E4M3, kernel_scale)
  return qkernel, new_kernel_scale

def input_qdq_bwd(res, g):
  qkernel_g = g
  new_kernel_scale = res
  return qkernel_g, new_kernel_scale

input_qdq.defvjp(input_qdq_fwd, input_qdq_bwd)

class TrainState(struct.PyTreeNode):
  """
  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    model_variables: The params that needs to be updated.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  model_variables: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    # For the variables in the params collection, we will use the optimizer as
    # usual.
    updates, new_opt_state = self.tx.update(
        grads['params'], self.opt_state, self.model_variables['params'])
    new_params = optax.apply_updates(self.model_variables['params'], updates)

    update_model_variables = core.unfreeze(self.model_variables)
    update_model_variables['params'] = new_params

    # For the fp8 variables in the fp8-params collection, we will simply replace
    # them with their grads, because their grads are actually new values defined
    # in the custom_vjp functions.
    if 'fp8_params' in grads:
      update_model_variables['fp8_params'] = grads['fp8_params']

    return self.replace(
        step=self.step + 1,
        model_variables=core.freeze(update_model_variables),
        opt_state=new_opt_state,
    )

  @classmethod
  def create(cls, apply_fn, model_variables, tx):
    """Creates a new instance with `step=0` and initialized `opt_state`."""

    params = model_variables['params']
    opt_state = tx.init(params)

    if 'fp8_params' in model_variables:
      fp8_params = model_variables['fp8_params']

    return cls(
        step=0,
        apply_fn=apply_fn,
        model_variables=model_variables,
        tx=tx,
        opt_state=opt_state,
    )

class DenseGeneralF8(nn.Module):
  features: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param(
        'kernel',
        self.kernel_init,
        (inputs.shape[1], self.features),
        self.param_dtype)
    scale_args = (
        nn.initializers.ones_init(),
        random.PRNGKey(0),
        (1,),
        jnp.float32,
    )
    kernel = jnp.asarray(kernel, self.dtype)
    input_scale = self.variable(
        'fp8_params',
        'input_scale',
        *scale_args,)
    kernel_scale = self.variable(
        'fp8_params',
        'kernel_scale',
        *scale_args,)

    inputs = input_qdq(inputs, input_scale.value)
    kernel = input_qdq(kernel, kernel_scale.value)
    out = jnp.dot(inputs, kernel)

    return out

class TwoDenses(nn.Module):
  hidden_size: int = 16

  def setup(self):
    self.densef8 = DenseGeneralF8(3 * self.hidden_size, dtype=dtype)
    self.dense = nn.DenseGeneral(self.hidden_size, dtype=dtype)
  def __call__(self, inputs):
    x = self.densef8(inputs)
    x = self.dense(x)
    return x

dim=16

def run():
  model = TwoDenses(dim)
  
  x = random.normal(random.PRNGKey(0), (dim, dim), dtype=dtype)
  dy = random.normal(random.PRNGKey(0), (dim, dim), dtype=dtype)
  k = random.PRNGKey(0)
  
  initialized_var = model.init(k, x)
  opt = optax.adam(learning_rate=0.0001)
  ts_args = {'tx': opt, 'apply_fn': model.apply}
  ts_args['model_variables' if use_fp8 else 'params'] = initialized_var
  state = TrainState.create(**ts_args)

  def step_fn(state, x, dy):
    def loss_fn(vars, x, dy):
      y = state.apply_fn(vars, x)
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)
  
    grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1])
    loss, grads = grad_fn(state.model_variables, x, dy)
    state = state.apply_gradients(grads=grads[0])
    return state, loss, grads
  
  train_step_fn = jax.jit(step_fn)
  # Train One step and update
  state, loss, grads = train_step_fn(state, x, dy)
  jax.debug.print('End of 1st train step with loss: {}', loss)

  w = state.model_variables['params']['dense']['kernel']
  g_w = grads[0]['params']['dense']['kernel']
  jax.debug.print('grad of non-fp8 dense kernel:\nmax: {}\nmin: {}\nabs_max:{}\nabs_min: {}', jnp.max(g_w), jnp.min(g_w), jnp.max(jnp.abs(g_w)), jnp.min(jnp.abs(g_w)))

run()
