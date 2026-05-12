# OpenCL backend (skeleton)

This folder contains an initial skeleton for the OpenCL backend used when
porting cgbn operators to OpenCL. It provides:

- `cgbn_opencl.h` — small API for context and program helpers.
- `impl_opencl.cpp` — basic implementations for creating a context/queue and
  building programs from source at runtime.
- `kernels/base.cl` — a couple of tiny kernels useful for smoke tests.

Next steps:

1. Integrate `cgbn_opencl.h` into the backend dispatching layer.
2. Add per-operator kernels and host wrappers that mirror CUDA semantics.
3. Add CMake targets or runtime program-loading in the host library.
