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


```bash
python test_train_state.py --cts # use customized train state
python test_train_state.py # use normal train state
```

| cts or not   |    time/step(ms)   |
|:-------------:|------:|
| `cts` |  19.59|
| `not cts` |  25.7 |

Note: the name of fp8_param should end with `fp8_meta` to be compatible with customized train state.
