
## 我的分析

对于N=4096bit -gpucurves 128刚好填满GPU

1. 如果曲线数量 (work groups) 不足以填满GPU，增大TPI (Work group size) 几乎没有效果。
2. 曲线数量刚好填满GPU，TPI16有明显提升32不再提升。
3. 曲线数量过多，大TPI性能下降。


## TPI=8

```sh
$env:ECM_OPENCL_TPI='8';
'(2^3919-1)' | .\build\Debug\ecm.exe -v -gpu -d 1 -sigma 3:12345678 -gpucurves 32 1e3 0
-gpucurves
32:  opencl_ecm_stage1 returned: 0 gputime=13384.1 ms
64:  opencl_ecm_stage1 returned: 0 gputime=13555.5 ms
96:  opencl_ecm_stage1 returned: 0 gputime=14100.4 ms
128: opencl_ecm_stage1 returned: 0 gputime=16000.8 ms
256: opencl_ecm_stage1 returned: 0 gputime=30736.7 ms
```

## TPI=16

```sh
$env:ECM_OPENCL_TPI='16';
-gpucurves
96:  opencl_ecm_stage1 returned: 0 gputime=14045 ms
128: opencl_ecm_stage1 returned: 0 gputime=14624.3 ms
256: opencl_ecm_stage1 returned: 0 gputime=31893 ms
```

## TPI=32

```sh
$env:ECM_OPENCL_TPI='32';
-gpucurves
96:  opencl_ecm_stage1 returned: 0 gputime=14088.7 ms
128: opencl_ecm_stage1 returned: 0 gputime=14607.9 ms
256: opencl_ecm_stage1 returned: 0 gputime=32465.9 ms
```

