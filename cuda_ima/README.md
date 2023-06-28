# How to reproduce.ZZZZ

This script is to show the CUDA illegal memory access error triggered by spmd + fp8. 
```bash
python cuda_ima.py
```

On 8-GPUs machine, you expect to see:
```
...
    loss, grads = pjit_step_fn(initialized_state, x,dy)
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: external/xla/xla/service/gpu/nccl_utils.cc:275: NCCL operation ncclCommInitRank(&comm, nranks, id, rank) failed: internal error: while running replica 0 and partit
ion 0 of a replicated computation (other replicas may have failed as well).
2023-06-28 22:23:07.035314: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:772] failed to free device memory at 0x7fe009a46c00; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
```
