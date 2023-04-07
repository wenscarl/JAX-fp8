# How to reproduce

## Fp16 + rank 3 input, bias is fused as side input(C).
```bash
CUBLASLT_LOG_LEVEL=4 TF_XLA_FLAGS="--tf_xla_auto_jit=2" python test_dense.py --dim3
```
[2023-04-07 04:57:13][cublasLt][8291][Trace][cublasLtMatmul] A=0X7F8EB6002300 Adesc=[type=R_16F rows=32 cols=16 ld=32] B=0X7F8EB6001B00 Bdesc=[type=R_16F rows=16 cols=64 ld=16] C=0X7F8EB6003200 Cdesc=[type=R_16F rows=32 cols=64 ld=32] D=0X7F8EB6003200 Ddesc=[type=R_16F rows=32 cols=64 ld=32] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F] algo=[algoId=21 tile=MATMUL_TILE_32x32 stages=MATMUL_STAGES_64x1] workSpace=0X0 workSpaceSizeInBytes=0 beta=? outOfPlace=0 stream=0X7EB04D0


## Fp16 + rank 2 input, bias is fused as epilog.
```bash
CUBLASLT_LOG_LEVEL=4 TF_XLA_FLAGS="--tf_xla_auto_jit=2" python test_dense.py
```
[2023-04-07 04:55:25][cublasLt][7643][Trace][cublasLtMatmul] A=0X7FB506002E80 Adesc=[type=R_16F rows=32 cols=16 ld=32] B=0X7FB506002A80 Bdesc=[type=R_16F rows=16 cols=32 ld=16] C=0X7FB506001300 Cdesc=[type=R_16F rows=32 cols=32 ld=32] D=0X7FB506001300 Ddesc=[type=R_16F rows=32 cols=32 ld=32] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F epilogue=**EPILOGUE_BIAS** biasPointer=0x7fb506003280] algo=[algoId=21 tile=MATMUL_TILE_32x32 stages=MATMUL_STAGES_64x1] workSpace=0X0 workSpaceSizeInBytes=0 beta=0 outOfPlace=0 stream=0X781AEA0



## Fp8 + rank 3 input, bias is not fused(Need to improve).
```bash
CUBLASLT_LOG_LEVEL=4 TF_XLA_FLAGS="--tf_xla_auto_jit=2" python test_dense.py --dim3 --fp8
```
[2023-04-07 04:56:59][cublasLt][8076][Trace][cublasLtMatmul] A=0X7EF80A005600 Adesc=[type=R_8F_E4M3 rows=32 cols=16 ld=16 order=ORDER_ROW] B=0X7EF80A005200 Bdesc=[type=R_8F_E4M3 rows=16 cols=64 ld=16] C=0X0 Cdesc=[type=R_16F rows=32 cols=64 ld=32] D=0X7EF80A004200 Ddesc=[type=R_16F rows=32 cols=64 ld=32] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F aScalePointer=0x7ef80a005800 bScalePointer=0x7ef80a005880 cScalePointer=0x7f0f4e800500 dScalePointer=0x7f0f4e800500] algo=[algoId=39 tile=MATMUL_TILE_256x64 stages=MATMUL_STAGES_128xAUTO] workSpace=0X7EF80A002000 workSpaceSizeInBytes=131 beta=0 outOfPlace=1 stream=0X8353620


## Fp8 + rank 2 input, bias is fused as epilog.
```bash
CUBLASLT_LOG_LEVEL=4 TF_XLA_FLAGS="--tf_xla_auto_jit=2" python test_dense.py --fp8
```
[2023-04-07 04:55:51][cublasLt][7861][Trace][cublasLtMatmul] A=0X7F969A002A00 Adesc=[type=R_8F_E4M3 rows=32 cols=16 ld=16 order=ORDER_ROW] B=0X7F969A002C00 Bdesc=[type=R_8F_E4M3 rows=16 cols=32 ld=16] C=0X0 Cdesc=[type=R_16F rows=32 cols=32 ld=32] D=0X7F969A001800 Ddesc=[type=R_16F rows=32 cols=32 ld=32] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F epilogue=**EPILOGUE_BIAS** biasPointer=0x7f969a002e00 aScalePointer=0x7f969a002f00 bScalePointer=0x7f969a002f80 cScalePointer=0x7fadde800500 dScalePointer=0x7fadde800500] algo=[algoId=39 tile=MATMUL_TILE_256x64 stages=MATMUL_STAGES_128xAUTO] workSpace=0X7F969A003300 workSpaceSizeInBytes=131 beta=0 outOfPlace=1 stream=0X84A9450

