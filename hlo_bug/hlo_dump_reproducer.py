import tensorflow as tf
from tensorflow.python.framework import dtypes

f8e4m3 = dtypes.float8_e4m3fn
f8e5m2 = dtypes.float8_e5m2
E4M3_max = 448.

# Create or load the wide-type data.
a = tf.random.uniform((16, 64), dtype=dtypes.float16)
b = tf.random.uniform((64, 16), dtype=dtypes.float16)

# Begin with factors of unity. The factors need to have the same type as the wide-type data.
# The first few training steps will warm up and adjust the scaling factors.
a_scale = tf.constant(1.0, dtypes.float16)
b_scale = tf.constant(1.0, dtypes.float16)
c_scale = tf.constant(1.0, dtypes.float16)

# Convert to FP8.
a_fp8 = tf.cast(a, f8e4m3)
b_fp8 = tf.cast(b, f8e4m3)

# JIT your model, which takes-in both the FP8 data along with scaling factors.
# Note that we now also pass the (delayed) scaling factor for the output.
@tf.function(jit_compile=True)
def matmul_fp8(a_fp8, a_scale, b_fp8, b_scale, c_scale):
    # Up-cast from FP8 to a wider type.
    a = tf.cast(a_fp8, dtypes.float16) * a_scale
    b = tf.cast(b_fp8, dtypes.float16) * b_scale

    # Call the GEMM operation.
    c = tf.matmul(a, b)

    # Quantize the output. These steps need to be followed exactly.
    # We clip before casting to ensure proper saturation and protect against overflow.
    saturated_c = tf.clip_by_value(c / c_scale, -E4M3_max, E4M3_max)
    c_fp8 = tf.cast(saturated_c, f8e4m3)
    new_c_scale = tf.reduce_max(tf.abs(c)) / E4M3_max

    # Return the new scaling factors along with the results.
    # The new scaling factors will be used in the next training step.
    return c_fp8, new_c_scale

c_fp8, c_scale = matmul_fp8(a_fp8, a_scale, b_fp8, b_scale, c_scale)

print(c_fp8)
print(c_scale)
hlo = matmul_fp8.experimental_get_compiler_ir(a_fp8, a_scale, b_fp8, b_scale, c_scale)('optimized_hlo')
print(hlo)
