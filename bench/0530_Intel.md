Intel 

```sh
C:\Users\SSRD_\Desktop\opencl-ecm>opencl_ecm_montsqr.exe -d 0 --bits 512 5000 128 2
OpenCL device override: CGBN_OPENCL_DEVICE_INDEX=0
ECM montgomery square microbench: 512-bit, kernel_iterations=5000, instances=128, launch_repeats=2, mode=wg, tpi=4
Note: mont_mul_asm bench skipped (AMD GPU only)
WG build opts: impl=4 impl4_unroll=2
OpenCL: compiling kernels (large source; may take minutes on NVIDIA)...
  [ecm_mont_mul_priv_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_opt_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_opt_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_unroll_only_512_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_unroll_only_512_manual_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_unroll_only_512_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_mul_priv_fips512_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_fips512_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_fips512_mt4_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_fips512_mt4_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_fips512_mt8_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_fips512_mt8_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_fips512_mt16_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_fips512_mt16_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_fips512_mt8_cs_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_sqr_priv_fips512_mt8_cs_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_mul_priv_fips512_mt16_cs_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_sqr_priv_fips512_mt16_cs_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_mul_priv_local_only_512_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_sqr_priv_local_only_512_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_mul_priv_opt2_512_local_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_sqr_priv_opt2_512_local_bench] private_mem=0B local_mem=0B pref_wg=16 max_wg=512
  [ecm_mont_mul_priv_unroll32_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_unroll32_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_unroll64_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_unroll64_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [sqr_unroll_only_512 vs mul(a,a)] MATCH
  [fips512 vs unroll_only_512] MATCH
  [fips512_mt4 vs fips512] MATCH
  [fips512_mt8 vs fips512] MATCH
  [fips512_mt16 vs fips512] MATCH
  [fips512_mt8_cs vs fips512] MATCH
  [fips512_mt16_cs vs fips512] MATCH
  [unroll_only_512 vs manual] MATCH
  [priv vs priv_opt mul] MATCH
  [priv vs priv_opt sqr] MATCH
mont_mul_priv:     3164.42 ms, 404498 ops/s
mont_mul_priv_opt: 3173.28 ms, 403368 ops/s (vs priv: 0.997209x)
mont_sqr_priv:     3156.26 ms, 405543 ops/s
mont_sqr_priv_opt: 4174.24 ms, 306643 ops/s (vs priv: 0.756129x)
mont_mul_priv_unroll_only_512: 865.22 ms, 1.47939e+06 ops/s (vs opt: 3.66759x)
mont_mul_priv_unroll_only_512_manual: 862.514 ms, 1.48403e+06 ops/s (vs unroll_only_512: 1.00314x)
mont_sqr_priv_unroll_only_512: 189.176 ms, 6.76619e+06 ops/s (vs opt: 22.0654x, vs mul_unroll_only_512: 4.57363x)
mont_mul_priv_fips512: 828.134 ms, 1.54564e+06 ops/s (vs unroll_only_512: 1.04478x)
mont_sqr_priv_fips512: 775.019 ms, 1.65157e+06 ops/s (vs fips512_mul: 1.06853x)
mont_mul_priv_fips512_mt4: 887.573 ms, 1.44214e+06 ops/s (vs fips512: 0.933032x)
mont_sqr_priv_fips512_mt4: 902.173 ms, 1.4188e+06 ops/s (vs fips512: 0.859059x)
mont_mul_priv_fips512_mt8: 827.482 ms, 1.54686e+06 ops/s (vs fips512: 1.00079x)
mont_sqr_priv_fips512_mt8: 839.522 ms, 1.52468e+06 ops/s (vs fips512: 0.923167x)
mont_mul_priv_fips512_mt16: 799.167 ms, 1.60167e+06 ops/s (vs fips512: 1.03625x)
mont_sqr_priv_fips512_mt16: 817.262 ms, 1.56621e+06 ops/s (vs fips512: 0.948312x)
mont_mul_priv_fips512_mt8_cs: 876.094 ms, 1.46103e+06 ops/s (vs fips512: 0.945257x)
mont_sqr_priv_fips512_mt8_cs: 893.208 ms, 1.43304e+06 ops/s (vs fips512: 0.867681x)
mont_mul_priv_fips512_mt16_cs: 895.277 ms, 1.42972e+06 ops/s (vs fips512: 0.925002x)
mont_sqr_priv_fips512_mt16_cs: 887.947 ms, 1.44153e+06 ops/s (vs fips512: 0.872822x)
mont_mul_priv_local_only_512:  238.375 ms, 5.3697e+06 ops/s (vs opt: 13.3121x)
mont_sqr_priv_local_only_512:  240.509 ms, 5.32205e+06 ops/s (vs opt: 17.3559x)
mont_mul_priv_opt2_512_local: 232.837 ms, 5.49742e+06 ops/s (vs opt: 13.6288x)
mont_sqr_priv_opt2_512_local: 235.999 ms, 5.42375e+06 ops/s (vs opt: 17.6875x)
mont_mul_priv_unroll32:  3372.64 ms, 379524 ops/s (vs opt: 0.940888x)
mont_sqr_priv_unroll32:  3372.57 ms, 379532 ops/s (vs opt: 1.2377x)
mont_mul_priv_unroll64:  4375.34 ms, 292549 ops/s (vs opt: 0.725265x)
mont_sqr_priv_unroll64:  3374.83 ms, 379278 ops/s (vs opt: 1.23687x)
  [cgbn_mont_mul_wg_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [cgbn_mont_sqr_wg_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
mont_mul_wg:   1189.61 ms, 1.07599e+06 ops/s
mont_sqr_wg:   1170.47 ms, 1.09357e+06 ops/s
  [cgbn_mont_mul_wg_bench] GMP verify: PASS
  [cgbn_mont_mul_wg_bench] GMP verify: PASS
  [cgbn_mont_mul_wg_bench b=a copy] GMP verify: PASS
  [cgbn_mont_sqr_wg_bench] GMP verify: PASS
```



```sh
C:\Users\SSRD_\Desktop\opencl-ecm>opencl_ecm_montsqr.exe -d 0 --bits 4096 100 128 2
OpenCL device override: CGBN_OPENCL_DEVICE_INDEX=0
ECM montgomery square microbench: 4096-bit, kernel_iterations=100, instances=128, launch_repeats=2, mode=wg, tpi=4
WG build opts: impl=4 impl4_unroll=2
OpenCL: compiling kernels (large source; may take minutes on NVIDIA)...
  [ecm_mont_mul_priv_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_opt_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_opt_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_unroll32_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_unroll32_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_unroll64_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_unroll64_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_unroll64_4096_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_unroll64_4096_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_unroll64_4096_nod_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_unroll64_4096_nod_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_unroll64_4096_mt2_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_unroll64_4096_mt2_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_unroll64_4096_mt2_weak_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_unroll64_4096_mt2_weak_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_unroll64_4096_mt4_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_sqr_priv_unroll64_4096_mt4_bench] private_mem=0B local_mem=0B pref_wg=32 max_wg=512
  [ecm_mont_mul_priv_unroll64_4096_mt8_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_unroll64_4096_mt8_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_fips4096_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_fips4096_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_fips4096_mt4_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_fips4096_mt4_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_fips4096_mt8_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_fips4096_mt8_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_fips4096_mt16_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_fips4096_mt16_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_fips4096_mt8_cs_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_fips4096_mt8_cs_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_mul_priv_fips4096_mt16_cs_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [ecm_mont_sqr_priv_fips4096_mt16_cs_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [sqr_unroll64_4096 vs mul(a,a)] MATCH
  [sqr_unroll64_4096_mt4 vs baseline] MATCH
  [sqr_unroll64_4096_mt8 vs baseline] MATCH
  [fips4096 vs unroll64_4096] MATCH
  [fips4096_mt4 vs fips4096] MATCH
  [fips4096_mt8 vs fips4096] MATCH
  [fips4096_mt16 vs fips4096] MATCH
  [fips4096_mt8_cs vs fips4096] MATCH
  [fips4096_mt16_cs vs fips4096] MATCH
  [priv vs priv_opt mul] MATCH
  [priv vs priv_opt sqr] MATCH
mont_mul_priv:     3595.56 ms, 7119.9 ops/s
mont_mul_priv_opt: 3511.55 ms, 7290.23 ops/s (vs priv: 1.02392x)
mont_sqr_priv:     4588.06 ms, 5579.7 ops/s
mont_sqr_priv_opt: 4505.65 ms, 5681.76 ops/s (vs priv: 1.01829x)
mont_mul_priv_unroll32:  1587.74 ms, 16123.5 ops/s (vs opt: 2.21166x)
mont_sqr_priv_unroll32:  1602.37 ms, 15976.3 ops/s (vs opt: 2.81186x)
mont_mul_priv_unroll64:  3371.27 ms, 7593.57 ops/s (vs opt: 1.04161x)
mont_sqr_priv_unroll64:  4358.89 ms, 5873.06 ops/s (vs opt: 1.03367x)
mont_mul_priv_unroll64_4096: 3990.66 ms, 6414.98 ops/s (vs generic64: 0.844791x)
mont_sqr_priv_unroll64_4096: 3989.38 ms, 6417.04 ops/s (vs generic64: 1.09262x, vs mul_unroll64_4096: 1.00032x)
mont_mul_priv_unroll64_4096_nod: 3989.69 ms, 6416.54 ops/s (vs unroll64_4096: 1.00024x)
mont_sqr_priv_unroll64_4096_nod: 3993.56 ms, 6410.31 ops/s (vs unroll64_4096: 0.998952x)
mont_mul_priv_unroll64_4096_mt2: 411.813 ms, 62164.1 ops/s (vs unroll64_4096: 9.69046x)
mont_sqr_priv_unroll64_4096_mt2: 419.126 ms, 61079.4 ops/s (vs unroll64_4096: 9.51832x)
mont_mul_priv_unroll64_4096_mt2_weak: 386.22 ms, 66283.4 ops/s (vs unroll64_4096: 10.3326x)
mont_sqr_priv_unroll64_4096_mt2_weak: 393.632 ms, 65035.4 ops/s (vs unroll64_4096: 10.1348x)
mont_mul_priv_unroll64_4096_mt4: 565.602 ms, 45261.5 ops/s (vs unroll64_4096: 7.0556x)
mont_sqr_priv_unroll64_4096_mt4: 572.365 ms, 44726.7 ops/s (vs unroll64_4096: 6.96998x)
mont_mul_priv_unroll64_4096_mt8: 1118.91 ms, 22879.4 ops/s (vs unroll64_4096: 3.56655x)
mont_sqr_priv_unroll64_4096_mt8: 1101.61 ms, 23238.8 ops/s (vs unroll64_4096: 3.62142x)
mont_mul_priv_fips4096: 987.625 ms, 25920.8 ops/s (vs unroll64_4096: 4.04067x)
mont_sqr_priv_fips4096: 988.414 ms, 25900.1 ops/s (vs fips4096_mul: 0.999202x)
mont_mul_priv_fips4096_mt4: 1023.33 ms, 25016.4 ops/s (vs fips4096: 0.965112x)
mont_sqr_priv_fips4096_mt4: 1025.44 ms, 24964.9 ops/s (vs fips4096: 0.963892x)
mont_mul_priv_fips4096_mt8: 920.668 ms, 27805.9 ops/s (vs fips4096: 1.07273x)
mont_sqr_priv_fips4096_mt8: 939.151 ms, 27258.7 ops/s (vs fips4096: 1.05245x)
mont_mul_priv_fips4096_mt16: 861.283 ms, 29723.1 ops/s (vs fips4096: 1.14669x)
mont_sqr_priv_fips4096_mt16: 864.801 ms, 29602.2 ops/s (vs fips4096: 1.14294x)
mont_mul_priv_fips4096_mt8_cs: 16526.8 ms, 1549 ops/s (vs fips4096: 0.0597588x)
mont_sqr_priv_fips4096_mt8_cs: 17926 ms, 1428.09 ops/s (vs fips4096: 0.0551386x)
mont_mul_priv_fips4096_mt16_cs: 29673.1 ms, 862.734 ops/s (vs fips4096: 0.0332835x)
mont_sqr_priv_fips4096_mt16_cs: 29767.8 ms, 859.99 ops/s (vs fips4096: 0.0332041x)
  [cgbn_mont_mul_wg_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
  [cgbn_mont_sqr_wg_bench] private_mem=0B local_mem=0B pref_wg=64 max_wg=512
mont_mul_wg:   1289.51 ms, 19852.5 ops/s
mont_sqr_wg:   1260.22 ms, 20314 ops/s
  [cgbn_mont_mul_wg_bench] GMP verify: PASS
  [cgbn_mont_mul_wg_bench] GMP verify: PASS
  [cgbn_mont_mul_wg_bench b=a copy] GMP verify: PASS
  [cgbn_mont_sqr_wg_bench] GMP verify: PASS
```