import tensorflow as tf
from tensorflow import summary
from dataclasses import dataclass
import optax
from flax import struct
from functools import partial
from typing import (Any, Callable, Iterable, List, Optional,
                    Mapping, Sequence, Tuple, Union)
import jax
import jax.numpy as jnp
import flax
from jax import lax
from flax import linen as nn
import re
from flax.traverse_util import flatten_dict, unflatten_dict
import argparse

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--fp8', action='store_true', help='use_fp8')
parser.add_argument('--train', action='store_true', help='train_with_state')
parser.add_argument('--scale', type=int, help='model_scale', default=1)

args = parser.parse_args()

model_size_scale = args.scale
use_fp8 = args.fp8
is_train = args.train
print("DEBUG: use_fp8", use_fp8)
print("DEBUG: model_scale", model_size_scale)
print("DEBUG: is_train", is_train)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
ActivationFn = Any

FAKE_E4M3 = jnp.float8_e4m3fn
FAKE_E5M2 = jnp.float8_e5m2
E4M3_MAX = 448
E5M2_MAX = 57344

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

def tree_shape(x): return jax.tree_map(lambda v: v.shape, x)

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
def in_qdq(input, in_scale, in_grad_scale, dummy=None):
  qin, new_in_scale = qdq_and_new_scale(input, FAKE_E4M3, in_scale)
  # input_grad_scale is needed in vjp
  return qin, new_in_scale, in_grad_scale

def in_qdq_fwd(input, in_scale, in_grad_scale, dummy):
  # new_in_grad_scale is a dummy value
  qin, new_in_scale, new_in_grad_scale = in_qdq(
      input, in_scale, in_grad_scale, dummy)
  return (qin, new_in_scale, new_in_grad_scale), (in_grad_scale, )

def in_qdq_bwd(res, g):
  in_grad_scale, = res
  qin_g, new_in_scale_g, in_grad_scale_g = g
  in_grad, new_in_grad_scale = qdq_and_new_scale(
      qin_g, FAKE_E5M2, in_grad_scale)
  return in_grad, jnp.zeros_like(new_in_scale_g), jnp.zeros_like(
      in_grad_scale_g), new_in_grad_scale


in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)

@jax.custom_vjp
def out_qdq(out, out_scale, out_grad_scale, dummy=None):
  # fwd do nothing
  # out_grad_scale is needed in vjp
  return out, out_scale, out_grad_scale

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


def initializer_32(): return jnp.array(32.0, dtype=jnp.float32)

class DenseWithScaling(nn.Module):
  features: int
  param_dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  activation: Optional[ActivationFn] = None
  use_quant: bool = False
  use_bias: bool = False
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
    out = jnp.dot(inputs, kernel)
#    if self.use_bias:
#        out = out + bias
    if self.activation:
      out = self.activation(out)

#    if self.use_quant and self.is_last:
#      output_scale = self.variable('qscale', 'output_scale', initializer_32)
#      output_grad_scale = self.variable(
#          'qscale', 'output_grad_scale', initializer_32)
#      # output_grad_scale is updated in training loop
#      output_grad_scale_perturb = self.variable(
#          'grad_qscale_placeholder', 'output_grad_scale_placeholder', initializer_32)
#      out, new_out_scale, new_out_grad_scale = out_qdq(
#          out, output_scale.value, output_grad_scale.value,
#          output_grad_scale_perturb.value)
#      output_scale.value = new_out_scale
    return out

def _convert_to_activation_function(
        fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  hidden_size: int = 2048
  ffn_hidden_size: int = 2048
  activations: Sequence[Union[str, Callable]] = ('gelu',)
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  dtype: Any = jnp.float32
  use_quant: bool = True

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseWithScaling(
          self.ffn_hidden_size,
          kernel_init=self.kernel_init,
          name=dense_name)(
              inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)
#    x = DenseWithScaling(self.ffn_hidden_size,use_quant=self.use_quant)(inputs)

    # Take elementwise product of above intermediate activations.
    # Apply dropout and final dense output projection.
    output = DenseWithScaling(
        self.hidden_size,
        use_quant=self.use_quant,
        name='wo')(
            x)
    return output

class DotProductAttention(nn.Module):
  """Attention operation in Transformer layer"""
  num_attention_heads: int
  kv_channels: int
  attention_dropout: float = 0.01

  def masked_softmax(
      self,
      inp: jnp.ndarray,
      mask: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    if mask is not None:
      inp = jnp.where(mask, -10000.0, inp)
    return jax.nn.softmax(inp, axis=-1)

  @nn.compact
  def __call__(self, query, key, value, attention_mask=None):
    b = query.shape[0]
    np = query.shape[2]
    sq = query.shape[1]
    sk = key.shape[1]
    hn = value.shape[3]

    # [b, sq, np, bn] -> [b, np, sq, bn]
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 3, 1))
    query = jnp.reshape(query, (b * np, sq, hn))
    key = jnp.reshape(key, (b * np, hn, sk))

    norm_factor = jnp.sqrt(float(self.kv_channels))
    bmm1 = jnp.matmul(query, key) / norm_factor

    # change view to [b, np, sq, sk]
    attention_scores = jnp.reshape(bmm1, (b, np, sq, sk))

    attention_probs = self.masked_softmax(
        attention_scores, None)  # attention_mask)

    attention_probs = nn.Dropout(
        rate=self.attention_dropout, deterministic=True)(attention_probs)

    attention_probs = self.masked_softmax(attention_scores, attention_mask)

    # change view [sk, b * np, hn]
    # value = jnp.reshape(value, (sk, b * np, hn))
    value = jnp.reshape(
        jnp.transpose(value, axes=(0, 2, 1, 3)),
        (b * np, sk, hn))

    # change view [b * np, sq, sk]
    attention_probs = jnp.reshape(attention_probs, (b * np, sq, sk))

    context = jnp.matmul(attention_probs, value)

    # change view to [b*np, sq, hn] - >[b, sq, np * hn]
    context = jnp.reshape(context, (b, np, sq, hn))
    context = jnp.transpose(context, axes=(0, 2, 1, 3))
    context = jnp.reshape(context, (b, sq, np * hn))
    return context

class BasicTransformer(nn.Module):
  use_quant: bool = False
  hidden_size: int = 512
  ffn_hidden_size: int = 1024
  num_attention_heads: int = 8
  layernorm_eps: float = 0.001
  attention_dropout: float = 0.01
  hidden_dropout: float = 0.01

  def setup(self):
    self.ln1 = nn.LayerNorm(epsilon=self.layernorm_eps)
    self.attention = DotProductAttention(
        num_attention_heads=self.num_attention_heads,
        kv_channels=self.hidden_size // self.num_attention_heads,
        attention_dropout=self.attention_dropout)
    self.dropout = nn.Dropout(self.hidden_dropout, deterministic=True)

    self.ln2 = nn.LayerNorm(epsilon=self.layernorm_eps)
    self.mlp = MlpBlock(hidden_size=self.hidden_size,
                        ffn_hidden_size=self.ffn_hidden_size,use_quant=self.use_quant)
    self.projection = DenseWithScaling(
        self.hidden_size, use_quant=self.use_quant, is_last=True)
    self.projection2 = DenseWithScaling(activation=jax.nn.relu,
        features=3*self.hidden_size, use_quant=self.use_quant)

    self.qkv_projection = DenseWithScaling(activation=jax.nn.relu,
        features=3 * self.hidden_size, use_quant=self.use_quant)

  def __call__(self, inputs):
    x = self.ln1(inputs)
    x = self.projection2(x)
    x = self.projection(x)
#    return x
#    res = inputs
#    x = self.ln1(inputs)
#    qkv = self.qkv_projection(x)
#    qkv_shape = qkv.shape
#    new_shape = tuple([qkv_shape[0], qkv_shape[1], self.num_attention_heads,
#                      3 * self.hidden_size // self.num_attention_heads])
#    qkv = jnp.reshape(qkv, new_shape)
#    q, k, v = jnp.split(qkv, 3, axis=-1)
#    x = self.attention(q, k, v)
#    x = self.projection(x)
#    x = self.dropout(x)
#    x = res + x
#    res = x
#    x = self.ln2(x)
#    x = self.mlp(x)
#    return x + res


class TrainState(struct.PyTreeNode):
  step: int
  params: Any
  grad_qscale_placeholder: Any
  qscale: Any
  opt_state: optax.OptState
  tx: optax.GradientTransformation = struct.field(pytree_node=False)

  @staticmethod
  def create(vars, tx):
    params = flax.core.unfreeze(vars['params'])
    opt_state = tx.init(params)
    grad_qscale_placeholder = flax.core.unfreeze(
        vars['grad_qscale_placeholder']) if 'grad_qscale_placeholder' in vars else None
    qscale = flax.core.unfreeze(vars['qscale']) if 'qscale' in vars else None
    return TrainState(0, params, grad_qscale_placeholder, qscale, opt_state, tx)

  def get_diff_vars(self):
    if self.grad_qscale_placeholder:
      return {'params': self.params, "grad_qscale_placeholder": self.grad_qscale_placeholder}
    return {'params': self.params}

  def get_nondiff_vars(self):
    if self.qscale:
      return {'qscale': self.qscale}
    return {}

def loss_fn(model, diff_vars, nondiff_vars, input_batch):
  logits, updated_nondiff_vars = model.apply(
      {**diff_vars, **nondiff_vars},
      input_batch['x'],
      mutable=['qscale','grad_qscale_placeholder'])
  batched_loss = optax.l2_loss(logits, input_batch['y'])
  return jnp.mean(batched_loss), updated_nondiff_vars

def step_fn(model, train_state, input_batch):
  bound_loss_fn = partial(loss_fn, model)
  grad_fn = jax.value_and_grad(bound_loss_fn, has_aux=True)
  (loss_val, updated_nondiff_vars), diff_vars_grads = grad_fn(
      train_state.get_diff_vars(), train_state.get_nondiff_vars(), input_batch)
  params_updates, updated_opt_state = train_state.tx.update(
      diff_vars_grads['params'], train_state.opt_state, params=train_state.params)
  updated_params = optax.apply_updates(train_state.params, params_updates)
  # Update train state
  new_qscale_vars = updated_nondiff_vars['qscale'] if 'qscale' in updated_nondiff_vars else None

  # Update qscale with grad_qscale_placeholder for gradient scale entries.
  if 'qscale' in updated_nondiff_vars:
    grad_qscale_vals = {
        tuple(re.sub(r'_placeholder$', '', '/'.join(k)).split('/')): v
        for k, v in flatten_dict(diff_vars_grads['grad_qscale_placeholder']).items()
    }
    flat_new_qscale_vars = flatten_dict(new_qscale_vars)
    flat_new_qscale_vars.update(grad_qscale_vals)
    new_qscale_vars = unflatten_dict(flat_new_qscale_vars)

  return train_state.replace(
      step=train_state.step + 1, params=updated_params, qscale=new_qscale_vars,
      opt_state=updated_opt_state), loss_val


batch_size = 16
epochs = 50

hidden_size = 512 * model_size_scale
ffn_hidden_size = 256 * model_size_scale
num_attention_heads = 8
sequence_length = 128
dropout_rate = 0.0

kdata = jax.random.PRNGKey(1001)
xkey, ykey, xekey, yekey = jax.random.split(kdata, 4)
x_train = jax.random.uniform(xkey, shape=(
    batch_size, sequence_length, hidden_size))
x_eval = jax.random.uniform(ykey, shape=(
    batch_size // 2, sequence_length, hidden_size))
y_train = jax.random.uniform(xekey, shape=(
    batch_size, sequence_length, hidden_size))
y_eval = jax.random.uniform(yekey, shape=(
    batch_size // 2, sequence_length, hidden_size))
train_size = x_train.shape[0]


LOG_DIR = './model_3'
def run(use_quant: bool, tb_label: str):
  root_k = jax.random.PRNGKey(123)
  init_k, subk = jax.random.split(root_k)
  model = BasicTransformer(
      use_quant=use_quant, hidden_size=hidden_size,
      ffn_hidden_size=ffn_hidden_size, num_attention_heads=num_attention_heads,
      attention_dropout=dropout_rate, hidden_dropout=dropout_rate)
  init_vars = model.init(init_k, x_train)
#  print(model.apply(init_vars,x_train, mutable=['qscale']), x_train.shape)
  tx = optax.sgd(learning_rate=0.01, momentum=0.1)
  train_state = TrainState.create(init_vars, tx)
  summary_writer = tf.summary.create_file_writer('%s/%s' % (LOG_DIR, tb_label))

  train_step = jax.jit(partial(step_fn, model))
  eval_loss_fn = jax.jit(partial(loss_fn, model))
  step = 0
  for epoch_i in range(epochs):
    num_steps = train_size // batch_size
    for i in range(num_steps):
      input_batch = {
          'x': x_train[i * batch_size: (i + 1) * batch_size],
          'y': y_train[i * batch_size: (i + 1) * batch_size]}
      train_state, train_loss = train_step(train_state, input_batch)
      # For debugging only, otherwise it slows down training
      with summary_writer.as_default(step=step):
        if train_state.qscale:
          # print(f'epoch={epoch_i}, step={i}, train_state.qscale={train_state.qscale}')
          # Monitor quantization scales
          for k, v in flatten_dict(train_state.qscale).items():
            tf.summary.scalar('/'.join(k), v)
        tf.summary.scalar('train_loss', train_loss)
      step += 1
    eval_input_batch = {'x': x_eval, 'y': y_eval}
    eval_loss, _ = eval_loss_fn(
        train_state.get_diff_vars(),
        train_state.get_nondiff_vars(),
        input_batch)
    with summary_writer.as_default(step=step):
      tf.summary.scalar('eval_loss', eval_loss)
    print(
        f'epoch={epoch_i}, step={i}, train_loss={train_loss}, eval_loss={eval_loss}')

if not is_train:
  def loss_ln1(model, params, x, y):
    lo = model.apply(params, x)
    return jnp.mean(optax.l2_loss(lo , y))
  
  
  #@jax.jit
  def update(model, params, x, y):
    bound_loss_fn1 = partial(loss_ln1, model)  
    gradient_fn = jax.value_and_grad(bound_loss_fn1, has_aux=False)
    loss, grads = gradient_fn(params, x, y)
    return loss, grads
  
  def myrun(use_quant=False):
      root_k = jax.random.PRNGKey(123)
      init_k, subk = jax.random.split(root_k)
      model = BasicTransformer(
        use_quant=use_quant, hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size, num_attention_heads=num_attention_heads,
        attention_dropout=dropout_rate, hidden_dropout=dropout_rate)
      init_vars = model.init(init_k, x_train)
      mmm = jax.jit(partial(update, model))
      loss, grad = mmm( init_vars, x_train, y_train)
      return loss, grad
  
  a, b = myrun()
  print(a, b)
else:
  run(use_quant=use_fp8, tb_label='fp8')


