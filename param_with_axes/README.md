# How to reproduce.

This script is to show the GPU idle time resulted by `param_with_axes`. The reproducer is default to `nn.Module.param`.
```bash
python param_with_axes_reproducer.py --axes # use partitioning.param_with_axes
python param_with_axes_reproducer.py # use nn.Module.param
```
The runtime is compared as followed(read from nsight system).



| how is it defined   |    time/step(ms)   |
|:-------------:|------:|
| `param_with_axes` |  44.3|
| `nn.Module.param` |  33.9 |
