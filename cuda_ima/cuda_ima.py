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

#from fp8layers.jax import DenseGeneral, TrainState
from flax.core.frozen_dict import FrozenDict

# Sharding related
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
from flax import struct, traverse_util, linen as nn
from flax.linen import spmd # Flax Linen SPMD.
from flax.linen import partitioning as flax_partitioning
from typing import (Any, Callable, Iterable, List, Optional, Mapping, Sequence, Tuple, Union)
from typing import Callable, Iterable, Optional, Dict, Union, Any, Tuple
from flax.linen import partitioning as nn_partitioning


parser = argparse.ArgumentParser(description='Benchmark a basic encoder layer')
parser.add_argument('--d', type=int, help='matrix dim')
args = parser.parse_args()

dim = args.d

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

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
def out_qdq(out, out_grad_scale):
  return out
def out_qdq_fwd(out,  out_grad_scale):
  qout = out_qdq(out,  out_grad_scale)
  return qout, out_grad_scale

def out_qdq_bwd(res, g):
  out_grad_scale = res
  qout_g = g
  out_grad, new_out_grad_scale = qdq_and_new_scale(qout_g, FAKE_E5M2, out_grad_scale)
  return out_grad, new_out_grad_scale

out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)

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


class DenseGeneral(nn.Module):
  features: int
  param_dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs):

    
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        (inputs.shape[1], self.features),
        self.param_dtype,
        axes=self.kernel_axes)

    output_grad_scale = jnp.full((1,), 1.01)
    inputs_scale = jnp.full((1,), 0.99)

    inputs = input_qdq(inputs, inputs_scale)
    out = jnp.dot(inputs, kernel)
    out = out_qdq(out, output_grad_scale)

    return out

def run():
  rules = (('batch', 'data'),)
  device_mesh = mesh_utils.create_device_mesh((2, 1))
  mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
  
  model = DenseGeneral(dim, kernel_axes=('hidden', 'mlp'))
  
  x = random.normal(random.PRNGKey(0), (dim, dim))
  dy = random.normal(random.PRNGKey(0), (dim, dim))
  k = random.PRNGKey(0)
  
  spmd.set_logical_axis_rules(rules)
  
  initialized_state = model.init(k, x)
  
  def loss_fn(state, x, dy):
    x = spmd.with_logical_constraint(x, ('batch', 'embed'))
    dy = spmd.with_logical_constraint(dy, ('batch', 'mlp'))

    y = model.apply(state, x)
    loss = y * dy.astype(y.dtype)
    return jnp.sum(loss)
  
  pjit_step_fn = pjit(
      jax.value_and_grad(loss_fn, argnums=[0, 1]),
  )
  
  with mesh:
    loss, grads = pjit_step_fn(initialized_state, x,dy)
  return loss, grads

print(run())
