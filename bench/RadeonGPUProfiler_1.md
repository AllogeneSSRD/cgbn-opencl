

## 原始CUDA CGBN版本

```sh
/mnt/d/code/GIMPS/gmp-ecm/gpu$ echo '(2^521-1)' | ./ecm -v -gpu -sigma 3:12345678 -gpucurves 128 1e4 0

Computing batch product (of 14447 bits) of primes up to B1=10000 took 0ms
GPU: Using device code targeted for architecture compile_80
GPU: Ptx version is 80
GPU: maxThreadsPerBlock = 896
GPU: numRegsPerThread = 68 sharedMemPerBlock = 0 bytes
Compiling custom kernel for 640 bits should be ~144% faster see README.gpu
Copying 81920 bytes of curves data to GPU
CGBN<1024, 8> running kernel<4 block x 256 threads> input number is 521 bits
Checkpoint autosave interval: 600000 ms (10.00 min)
Computing 200 bits/call, 1/14447 (0.0%)
Computing 220 bits/call, 201/14447 (1.4%)
Computing 242 bits/call, 421/14447 (2.9%)
Computing 514 bits/call, 3177/14447 (22.0%)
Computing 1326 bits/call, 11346/14447 (78.5%)
Copying results back to CPU ...
Computing 128 Step 1 took 121ms of CPU time / 453ms of GPU time
```


## opencl wg

#### Dispatch properties
Total work groups {128，0，0}
Work group size {8，1，1}
Global dispatch size{1024，0，0}
#### Wavefronts and threads
Total wavefronts 128
Total threads 1,024
Average wavefront duration 3,322.450 μs
Average threads per wavefront 8
Wavefront mode wave32
#### Per-wavefront resources
Vector registers 19 (32 allocated)
Scalar registers 82 (128 allocated)
Registers spilled to scratch memory OFF
Local data share per thread group  2048bytes


The occupancy of this shader is not limited by any resources.
This shader could potentially run 16 wavefronts out of 16 wavefronts per SIMD.
You are already running maximum number of wavefronts for your ASIC.

```sh
PS D:\code\MPA-OpenCl>'(2^521-1)' | .\build\Debug\ecm.exe -v -gpu -d 1 -sigma 3:12345678 -gpucurves 128 1e4 0

[2026-05-21 10:00:27] ecm driver starting
[2026-05-21 10:00:27]   mode: gpu, gpucurves=128, gpuckpt_ms=600000, device=1
[2026-05-21 10:00:27]   B1=1e4, B2=0
[2026-05-21 10:00:27] Parsed N bit-size: 521
[2026-05-21 10:00:27] batch_s bit-size: 14447
[2026-05-21 10:00:27] Available OpenCL devices:
[2026-05-21 10:00:27]   [0] NVIDIA CUDA | NVIDIA GeForce RTX 4060 Laptop GPU | GPU | OpenCL 3.0 CUDA
[2026-05-21 10:00:27]   [1] AMD Accelerated Parallel Processing | gfx1150 | GPU | OpenCL 2.0 AMD-APP (3640.0)
[2026-05-21 10:00:29] GPU: will use device 1: gfx1150, OpenCL 2.0 AMD-APP (3640.0), 8 compute units.
[2026-05-21 10:00:29] GPU: driver 3640.0 (PAL,LC)
[2026-05-21 10:00:29] GPU: maxSharedPerBlock = 65536 maxThreadsPerBlock = 256 maxMemAllocPerBuffer = 2060669747
[2026-05-21 10:00:29] GPU: Selection and initialization of the device took 1417ms
[2026-05-21 10:00:29] OpenCL: built kernel MAX_LIMBS=32 (1417ms)
[2026-05-21 10:00:29] Using B1=10000, B2=0, sigma=3:12345678-12345805 (128 curves)
[2026-05-21 10:00:29] GPU: CGBN<1024> kernel, 521-bit N, 128 curves, sigma=12345678-12345805, s=14447 bits, np0=0x00000001
[2026-05-21 10:00:29] GPU: Computing 200 bits/call, 1/14447 (0.0%)
[2026-05-21 10:00:29] GPU: Computing 180 bits/call, 201/14447 (1.4%)
[2026-05-21 10:00:29] GPU: Computing 162 bits/call, 381/14447 (2.6%)
[2026-05-21 10:00:30] GPU: Computing 130 bits/call, 1558/14447 (10.8%)
[2026-05-21 10:00:31] GPU: Computing 130 bits/call, 2858/14447 (19.8%)
[2026-05-21 10:00:39] GPU: Computing 117 bits/call, 12335/14447 (85.4%)
[2026-05-21 10:00:40] opencl_ecm_stage1 returned: 0 gputime=11127.1 ms
```


### opencl no wg

#### Dispatch properties
Total work groups{128，0，0}
Work group size{128，1，1}
Global dispatch size{16384，0，0}
#### Wavefronts and threads
Total wavefronts 4
Total threads 128
Average wavefront duration 929.628 μs
Average threads per wavefront 32
Wavefront mode wave32
#### Per-wavefront resources
Vector registers 40 (48 allocated)
Scalar registers 36 (128 allocated)
Registers spilled to scratch memory OFF
Local data share per thread group 0

The occupancy of this shader is not limited by any resources.
This shader could potentially run 16 wavefronts out of 16 wavefronts per SIMD.
You are already running maximum number of wavefronts for your ASIC.

```sh
PS D:\code\MPA-OpenCl> '(2^521-1)' | .\build\Debug\ecm.exe -v -gpu -d 1 -sigma 3:12345678 -gpucurves 128 1e4 0

[2026-05-21 10:14:05] ecm driver starting
[2026-05-21 10:14:05]   mode: gpu, gpucurves=128, gpuckpt_ms=600000, device=1
[2026-05-21 10:14:05]   B1=1e4, B2=0
[2026-05-21 10:14:05] Parsed N bit-size: 521
[2026-05-21 10:14:05] batch_s bit-size: 14447
[2026-05-21 10:14:06] Available OpenCL devices:
[2026-05-21 10:14:06]   [0] NVIDIA CUDA | NVIDIA GeForce RTX 4060 Laptop GPU | GPU | OpenCL 3.0 CUDA
[2026-05-21 10:14:06]   [1] AMD Accelerated Parallel Processing | gfx1150 | GPU | OpenCL 2.0 AMD-APP (3640.0)
[2026-05-21 10:14:07] GPU: will use device 1: gfx1150, OpenCL 2.0 AMD-APP (3640.0), 8 compute units.
[2026-05-21 10:14:07] GPU: driver 3640.0 (PAL,LC)
[2026-05-21 10:14:07] GPU: maxSharedPerBlock = 65536 maxThreadsPerBlock = 256 maxMemAllocPerBuffer = 2060669747
[2026-05-21 10:14:07] GPU: Selection and initialization of the device took 1410ms
[2026-05-21 10:14:07] OpenCL: built kernel MAX_LIMBS=32 (1410ms)
[2026-05-21 10:14:07] Using B1=10000, B2=0, sigma=3:12345678-12345805 (128 curves)
[2026-05-21 10:14:07] GPU: CGBN<1024> kernel, 521-bit N, 128 curves, sigma=12345678-12345805, s=14447 bits, np0=0x00000001
[2026-05-21 10:14:07] GPU: Computing 200 bits/call, 1/14447 (0.0%)
[2026-05-21 10:14:08] GPU: Computing 180 bits/call, 201/14447 (1.4%)
[2026-05-21 10:14:08] GPU: Computing 162 bits/call, 381/14447 (2.6%)
[2026-05-21 10:14:10] GPU: Computing 100 bits/call, 1340/14447 (9.3%)
[2026-05-21 10:14:12] GPU: Computing 100 bits/call, 2340/14447 (16.2%)
[2026-05-21 10:14:26] GPU: Computing 100 bits/call, 10340/14447 (71.6%)
[2026-05-21 10:14:34] opencl_ecm_stage1 returned: 0 gputime=26201.1 ms
```

