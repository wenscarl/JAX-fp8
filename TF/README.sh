# TF benchamrk

```bash
CUBLASLT_LOG_LEVEL=0 TF_CPP_VMODULE=gemm_rewriter=1 TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_enable_cublaslt=true" python tf_benchmark.py --train --scale 4 --fp8
```

In 22.12 container with tf-nightly, 124ms/step(fp8 + fp32) vs 224ms/step(fp32); 99ms/step(fp8 + mixed) vs 117ms/step(mixed).