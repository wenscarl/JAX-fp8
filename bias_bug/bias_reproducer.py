import functools
import os
import tensorflow as tf

from contextlib import contextmanager
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.keras import layers, initializers, optimizers

import argparse

from DenseFp8 import Dense

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--fp8', action='store_true', help='mixed_precision')
parser.add_argument('--dim3', action='store_true', help='mixed_precision')
args = parser.parse_args()

use_fp8 = args.fp8
use_dim3 = args.dim3

@contextmanager
def disable_mixed_precision(disable=True):
  try:
    if disable:
      tf.keras.mixed_precision.set_global_policy('float32')
    yield
  finally:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

def roll_and_update(amax_h, update):
  amax_h = tf.roll(amax_h, shift=-1, axis=0)
  amax_h = tf.tensor_scatter_nd_update(amax_h, [[0]], [update])
  return amax_h

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  exp = tf.math.floor(tf.experimental.numpy.log2(fp8_max / amax)) - margin
  sf = tf.math.round(tf.math.pow(2., tf.math.abs(exp)))
  sf = tf.where(amax > 0.0, sf, scale)
  sf = tf.where(tf.math.is_finite(amax), sf, scale)
  sf = tf.where(exp < 0, 1.0 / sf, sf)
  return sf

def testDenseFwd():
  x_shape = (4, 16, 16) if use_dim3 else (32, 16)
  x = tf.random.uniform(x_shape, dtype=tf.float32)
  init = initializers.RandomUniform(minval=0., maxval=1.)

  dense_kwargs = {
      "units": 32,
      "use_bias": True,
      "kernel_initializer": init,
      "bias_initializer": init,
  }

  if use_fp8:
    dense = Dense(**dense_kwargs)
  else:
    dense = layers.Dense(**dense_kwargs)

  def _infer_step(x, model):
    return model(x)

  fn = functools.partial(_infer_step, model=dense)

  y = tf.function(fn, jit_compile=True)(x)
  print(y[0, 0])

tf.keras.mixed_precision.set_global_policy('mixed_float16')
testDenseFwd()
