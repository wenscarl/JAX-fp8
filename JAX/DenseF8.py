from typing import (Any, Callable, Iterable, List, Mapping, Optional, Sequence,
                            Tuple, Union)

import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

FAKE_E4M3 = jnp.float8_e4m3fn
FAKE_E5M2 = jnp.float8_e5m2
E4M3_MAX = 448
E5M2_MAX = 57344

# Type annotations
Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
ActivationFn = Callable[..., Array]
# Parameter initializers.


def initializer_32(): return jnp.array(32.0, dtype=jnp.float32)

def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  elif fake_dtype == FAKE_E5M2:
    return E5M2_MAX
  else:
    raise ValueError('Only FAKE_E4M3 and FAKE_E5M2 supported')

def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  scaled_x = jnp.clip(x / scale, -dtype_max, dtype_max)
  return scaled_x.astype(quantized_dtype)

def dequantize(x, wide_dtype, scale):
  return x.astype(wide_dtype) * scale

def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def compute_new_scale(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  amax = jnp.max(jnp.abs(x)).astype(scale.dtype)
  # Ensure scale != 0 and avoid divide-by-zero.
  amax = jnp.maximum(amax, 2**-10)
  return 1.1 * amax / dtype_max

def qdq_and_new_scale(x, dtype, scale):
  qx = quantize_dequantize(x, dtype, scale)
  new_scale = compute_new_scale(x, dtype, scale)
  return qx, new_scale
@jax.custom_vjp
def kernel_qdq(kernel, kernel_scale):
  qkernel, new_kernel_scale = qdq_and_new_scale(kernel, FAKE_E4M3, kernel_scale)
  return qkernel, new_kernel_scale

def kernel_qdq_fwd(kernel, kernel_scale):
  return kernel_qdq(kernel, kernel_scale), None

def kernel_qdq_bwd(_, g):
  # pass through gradients
  return g


kernel_qdq.defvjp(kernel_qdq_fwd, kernel_qdq_bwd)

@jax.custom_vjp
def in_qdq(inp, inp_scale, inp_grad_scale, dummy=None):
  qin, new_inp_scale = qdq_and_new_scale(inp, FAKE_E4M3, inp_scale)
  # out_grad_scale is needed in vjp
  return qin, new_inp_scale, inp_grad_scale

def inp_qdq_fwd(inp, inp_scale, inp_grad_scale, dummy):
  # new_inp_grad_scale is a dummy value
  qin, new_inp_scale, new_inp_grad_scale = in_qdq(
      inp, inp_scale, inp_grad_scale, dummy)
  return (qin, new_inp_scale, new_inp_grad_scale), (inp_grad_scale, )

def inp_qdq_bwd(res, g):
#  inp_grad_scale, = res
#  return g, inp_grad_scale
  inp_grad_scale, = res
  qin_g, new_inp_scale_g, inp_grad_scale_g = g
#  inp_grad, new_inp_grad_scale = qdq_and_new_scale(
#      qin_g, FAKE_E5M2, inp_grad_scale)
  return qin_g, new_inp_scale_g, inp_grad_scale_g, inp_grad_scale
#  return inp_grad, jnp.zeros_like(new_inp_scale_g), jnp.zeros_like(
#      inp_grad_scale_g), new_inp_grad_scale


in_qdq.defvjp(inp_qdq_fwd, inp_qdq_bwd)


@jax.custom_vjp
def out_qdq(out, out_scale, out_grad_scale, dummy=None):
  return out, out_scale, out_grad_scale
#  qout, new_out_scale = qdq_and_new_scale(out, FAKE_E4M3, out_scale)
  # out_grad_scale is needed in vjp
#  return qout, new_out_scale, out_grad_scale

def out_qdq_fwd(out, out_scale, out_grad_scale, dummy):
  # new_out_grad_scale is a dummy value
  qout, new_out_scale, new_out_grad_scale = out_qdq(
      out, out_scale, out_grad_scale, dummy)
  return (qout, new_out_scale, new_out_grad_scale), (out_grad_scale, )

def out_qdq_bwd(res, g):
  out_grad_scale, = res
  qout_g, new_out_scale_g, out_grad_scale_g = g
  out_grad, new_out_grad_scale = qdq_and_new_scale(
      qout_g, FAKE_E5M2, out_grad_scale)
  return out_grad, jnp.zeros_like(new_out_scale_g), jnp.zeros_like(
      out_grad_scale_g), new_out_grad_scale

out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)

class DenseWithScaling(nn.Module):
  features: int
  param_dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  activation: Optional[ActivationFn] = None
  use_quant: bool = False
  is_last: bool = False

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel', self.kernel_init,
                        (inputs.shape[-1], self.features), self.param_dtype)
    bias = self.param(
        'bias', self.bias_init, (self.features,),
        self.param_dtype)

    if self.use_quant:
      kernel_scale = self.variable('qscale', 'kernel_scale', initializer_32)
      kernel, new_kernel_scale = kernel_qdq(kernel, kernel_scale.value)
      kernel_scale.value = new_kernel_scale

      input_scale = self.variable('qscale', 'input_scale', initializer_32)
      input_grad_scale = self.variable(
          'qscale', 'input_grad_scale', initializer_32)
      input_grad_scale_perturb = self.variable(
          'grad_qscale_placeholder', 'input_grad_scale_placeholder', initializer_32)
      # input_grad_scale is updated in training loop
      inputs, new_input_scale, new_input_grad_scale = in_qdq(
          inputs, input_scale.value, input_grad_scale.value,
          input_grad_scale_perturb.value)
      input_scale.value = new_input_scale


    # Actual dense layer math.
    out = jnp.dot(inputs, kernel) + bias
    if self.activation:
      out = self.activation(out)

    if self.use_quant:
      output_scale = self.variable('qscale', 'output_scale', initializer_32)
      output_grad_scale = self.variable(
          'qscale', 'output_grad_scale', initializer_32)
      # output_grad_scale is updated in training loop
      output_grad_scale_perturb = self.variable(
          'grad_qscale_placeholder', 'output_grad_scale_placeholder', initializer_32)
      out, new_out_scale, new_out_grad_scale = out_qdq(
          out, output_scale.value, output_grad_scale.value,
          output_grad_scale_perturb.value)
      output_scale.value = new_out_scale
    return out

