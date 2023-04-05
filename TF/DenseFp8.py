# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the Dense layer."""


import tensorflow.compat.v2 as tf

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.input_spec import InputSpec

# isort: off
from tensorflow.python.util.tf_export import keras_export

from tensorflow.python.framework import dtypes

FAKE_E4M3 = dtypes.float8_e4m3fn
FAKE_E5M2 = dtypes.float8_e5m2

E4M3_MAX = 448.
E5M2_MAX = 57344.
AMAX_HIS_LEN = 16

def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  else:
    assert fake_dtype == FAKE_E5M2
    return E5M2_MAX

def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  policy = tf.keras.mixed_precision.global_policy()
  is_mixed_policy = (
      policy is not None and policy.compute_dtype != policy.variable_dtype
  )

  scaled_x = x / (tf.cast(scale, tf.float16) if is_mixed_policy else scale)
  clipped_x = tf.clip_by_value(scaled_x, -dtype_max, dtype_max)
  return tf.cast(clipped_x, quantized_dtype)

def dequantize(x, wide_dtype, scale):
  return tf.cast(x, wide_dtype) * tf.cast(scale, wide_dtype)

def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def update_scale(x, quantized_dtype, scale_var, amax_history):
  dtype_max = get_fp8_max(quantized_dtype)

  amax_current = tf.cast(tf.math.reduce_max(tf.math.abs(x)), scale_var.dtype)

  amax_his_tsr = tf.tensor_scatter_nd_update(
      tf.roll(amax_history.read_value(), 1, 0), [[0]], [amax_current])
  amax_history.assign(tf.cast(amax_his_tsr, amax_history.dtype))

  amax_temp = tf.reduce_max(amax_history, axis=0)
  amax = tf.maximum(amax_temp, 2 ** -10)
  scale_var.assign(tf.cast(1.1 * amax / dtype_max, scale_var.dtype))

def qdq_and_update(x, dtype, scale_var, amax_history):
  qx = quantize_dequantize(x, dtype, scale_var)
  update_scale(x, dtype, scale_var, amax_history)
  return qx

class DenseFp8(tf.keras.layers.Dense):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). These are all attributes of
    `Dense`.
    Note: If the input to the layer has a rank greater than 2, then `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors).  The output in this case will have
    shape `(batch_size, d0, units)`.
    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    When a popular kwarg `input_shape` is passed, then keras will create
    an input layer to insert before the current layer. This can be treated
    equivalent to explicitly defining an `InputLayer`.
    Example:
    >>> # Create a `Sequential` model and add a Dense layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(tf.keras.layers.Dense(32))
    >>> model.output_shape
    (None, 32)
    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        input_from_keras_layer=True,
        output_to_keras_layer=True,
        **kwargs,
    ):
      super().__init__(units, **kwargs)
      self.input_from_keras_layer = input_from_keras_layer
      self.output_to_keras_layer = output_to_keras_layer

    def build(self, input_shape):
      super().build(input_shape)

      init1 = tf.keras.initializers.Ones()
      init0 = tf.keras.initializers.Zeros()
      self.input_amax_history = self.add_weight(
          "input_amax_history", shape=(AMAX_HIS_LEN,), initializer=init0,
          trainable=False)
      self.input_scale = self.add_weight(
          "input_scale", shape=(), initializer=init1, trainable=False)

      self.kernel_amax_history = self.add_weight(
          "kernel_amax_history", shape=(AMAX_HIS_LEN,), initializer=init0,
          trainable=False)
      self.kernel_scale = self.add_weight(
          "kernel_scale", shape=(), initializer=init1, trainable=False)

      self.input_grad_amax_history = self.add_weight(
          "input_grad_amax_history", shape=(AMAX_HIS_LEN,),
          initializer=init0, trainable=False)
      self.input_grad_scale = self.add_weight(
          "input_grad_scale", shape=(), initializer=init1, trainable=False)

      self.output_grad_amax_history = self.add_weight(
          "output_grad_amax_history", shape=(AMAX_HIS_LEN,),
          initializer=init0, trainable=False)
      self.output_grad_scale = self.add_weight(
          "output_grad_scale", shape=(),
          initializer=init1, trainable=False)

      self.built = True

    @tf.custom_gradient
    def in_qdq(self, inp):
      """Quantize-dequantize the input and its gradient."""
      qin = qdq_and_update(inp, FAKE_E4M3, self.input_scale,
                           self.input_amax_history)

      def grad(in_grad):
        return in_grad

      return qin, grad

    @tf.custom_gradient
    def out_qdq(self, out):
      """Quantize-dequantize the output and its gradient"""

      def grad(out_grad):
          if self.output_to_keras_layer:
              return qdq_and_update(out_grad, FAKE_E5M2, self.output_grad_scale,
                                    self.output_grad_amax_history)
          return out_grad
      return out, grad

    @tf.custom_gradient
    def kernel_qdq(self, kernel):
      """Quantize-dequantize the kernel but not its gradient."""

      qkernel = qdq_and_update(kernel, FAKE_E4M3, self.kernel_scale,
                               self.kernel_amax_history)
      if self.is_mixed_policy:
        qkernel = tf.cast(qkernel, tf.float16)

      def grad(kernel_grad, variables=None):
        return kernel_grad

      return qkernel, grad

    def call(self, inputs):
      if (isinstance(inputs, tf.RaggedTensor) or
          isinstance(inputs, tf.SparseTensor)):
        return super().call(inputs)
        
      if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
        inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

      policy = tf.keras.mixed_precision.global_policy()
      self.is_mixed_policy = (
          policy is not None and policy.compute_dtype != policy.variable_dtype
      )
      
      rank = inputs.shape.rank
      if rank == 2 or rank is None:
        outputs = tf.matmul(a=self.in_qdq(inputs),
                            b=self.kernel_qdq(self.kernel))
      # TODO(shuw): originally, the keras uses the tensordot. But now we use
      # the matmul instead. Do we still care?
      # Broadcast kernel to inputs.
      else:
        outputs = tf.matmul(a=self.in_qdq(inputs),
                            b=self.kernel_qdq(self.kernel))
        # Reshape the output back to the original ndim of the input.
        if not tf.executing_eagerly():
          shape = inputs.shape.as_list()
          output_shape = shape[:-1] + [self.kernel.shape[-1]]
          outputs.set_shape(output_shape)

      if self.use_bias:
        outputs = tf.nn.bias_add(outputs, self.bias)

      if self.activation is not None:
        outputs = self.activation(outputs)

      outputs = self.out_qdq(outputs)

      return outputs

    def get_config(self):
      config = super().get_config()
      config.update(
          {
              "input_from_keras_layer": self.input_from_keras_layer,
              "output_to_keras_layer": self.output_to_keras_layer,
          }
      )
      return config
