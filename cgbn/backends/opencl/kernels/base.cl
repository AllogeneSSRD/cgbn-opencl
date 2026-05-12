// Minimal OpenCL kernels for cgbn backend testing

__kernel void test_add(__global int *a, __global int *b) {
    int gid = get_global_id(0);
    a[gid] = a[gid] + b[gid];
}

__kernel void test_copy(__global const int *src, __global int *dst) {
    int gid = get_global_id(0);
    dst[gid] = src[gid];
}
