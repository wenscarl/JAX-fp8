# How to reproduce.

```bash
CUBLASLT_LOG_LEVEL=5 python hlo_dump_reproducer.py
```

From CUBLASLT log, D is E4M3, but optimized HLO shows the return type of custom-call is f16[16,16].

