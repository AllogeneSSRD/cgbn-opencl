#ifndef HAS_ASM
#define HAS_ASM 1
#endif

#ifndef SELFTEST_DISABLE_MAD_U64
#define SELFTEST_DISABLE_MAD_U64 1
#endif

typedef long i64;
typedef ulong u64;
typedef uint u32;

#define KERNEL(x) __kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

static inline i64 run_latency_case(int what, int iterations) {
  i64 sink = 0;
  if (what == 0) { // V_NOP
    for (int i = 0; i < iterations; ++i) {
      __asm("v_nop");
    }
    sink = iterations;
  } else if (what == 1) { // V_ADD_I32
    int a = 2, b = 3;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_add_i32 %0, %1, %0" : "+v"(a) : "v"(b));
    }
    sink = (i64)a;
  } else if (what == 2) { // V_FMA_F32
    float a = 2.0f, b = 3.0f;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_fma_f32 %0, %0, %1, %0" : "+v"(a) : "v"(b));
    }
    sink = (i64)as_int(a);
  } else if (what == 3) { // V_ADD_F64
    double a = 2.0, b = 3.0;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_add_f64 %0, %0, %1" : "+v"(a) : "v"(b));
    }
    sink = (i64)as_long(a);
  } else if (what == 4) { // V_FMA_F64
    double a = 2.0, b = 3.0;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_fma_f64 %0, %0, %1, %0" : "+v"(a) : "v"(b));
    }
    sink = (i64)as_long(a);
  } else if (what == 5) { // V_MUL_F64
    double a = 2.0, b = 3.0;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_mul_f64 %0, %0, %1" : "+v"(a) : "v"(b));
    }
    sink = (i64)as_long(a);
  } else if (what == 6) { // V_MAD_U64_U32
#if SELFTEST_DISABLE_MAD_U64
    sink = -1;
#else
    u32 a = 2;
    u64 b = 3;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_mad_u64_u32 %1, vcc, %0, %0, %1" : : "v"(a), "v"(b));
    }
    sink = (i64)b;
#endif
  }
  return sink;
}

static inline i64 run_throughput_case(int what, int iterations) {
  i64 sink = 0;
  if (what == 0) { // V_NOP
    for (int i = 0; i < iterations; ++i) {
      __asm("v_nop\n\tv_nop\n\tv_nop\n\tv_nop");
    }
    sink = iterations * 4;
  } else if (what == 1) {
    int a0 = 2, a1 = 3, a2 = 4, a3 = 5, b = 7;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_add_i32 %0, %4, %0\n\tv_add_i32 %1, %4, %1\n\tv_add_i32 %2, %4, %2\n\tv_add_i32 %3, %4, %3"
            : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3) : "v"(b));
    }
    sink = (i64)(a0 + a1 + a2 + a3);
  } else if (what == 2) {
    float a0 = 2, a1 = 3, a2 = 4, a3 = 5, b = 7;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_fma_f32 %0, %0, %4, %0\n\tv_fma_f32 %1, %1, %4, %1\n\tv_fma_f32 %2, %2, %4, %2\n\tv_fma_f32 %3, %3, %4, %3"
            : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3) : "v"(b));
    }
    sink = (i64)(as_int(a0) ^ as_int(a1) ^ as_int(a2) ^ as_int(a3));
  } else if (what == 3) {
    double a0 = 2, a1 = 3, a2 = 4, a3 = 5, b = 7;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_add_f64 %0, %0, %4\n\tv_add_f64 %1, %1, %4\n\tv_add_f64 %2, %2, %4\n\tv_add_f64 %3, %3, %4"
            : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3) : "v"(b));
    }
    sink = (i64)(as_long(a0) ^ as_long(a1) ^ as_long(a2) ^ as_long(a3));
  } else if (what == 4) {
    double a0 = 2, a1 = 3, a2 = 4, a3 = 5, b = 7;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_fma_f64 %0, %0, %4, %0\n\tv_fma_f64 %1, %1, %4, %1\n\tv_fma_f64 %2, %2, %4, %2\n\tv_fma_f64 %3, %3, %4, %3"
            : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3) : "v"(b));
    }
    sink = (i64)(as_long(a0) ^ as_long(a1) ^ as_long(a2) ^ as_long(a3));
  } else if (what == 5) {
    double a0 = 2, a1 = 3, a2 = 4, a3 = 5, b = 7;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_mul_f64 %0, %0, %4\n\tv_mul_f64 %1, %1, %4\n\tv_mul_f64 %2, %2, %4\n\tv_mul_f64 %3, %3, %4"
            : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3) : "v"(b));
    }
    sink = (i64)(as_long(a0) ^ as_long(a1) ^ as_long(a2) ^ as_long(a3));
  } else if (what == 6) {
#if SELFTEST_DISABLE_MAD_U64
    sink = -1;
#else
    u32 a = 2;
    u64 b0 = 3, b1 = 5, b2 = 7, b3 = 11;
    for (int i = 0; i < iterations; ++i) {
      __asm("v_mad_u64_u32 %1, vcc, %0, %0, %1" : : "v"(a), "v"(b0));
      __asm("v_mad_u64_u32 %1, vcc, %0, %0, %1" : : "v"(a), "v"(b1));
      __asm("v_mad_u64_u32 %1, vcc, %0, %0, %1" : : "v"(a), "v"(b2));
      __asm("v_mad_u64_u32 %1, vcc, %0, %0, %1" : : "v"(a), "v"(b3));
    }
    sink = (i64)(b0 ^ b1 ^ b2 ^ b3);
#endif
  }
  return sink;
}

KERNEL(64) testLatency(int what, int iterations, __global i64 *io) {
#if HAS_ASM
  size_t gid = get_global_id(0);
  io[gid] = run_latency_case(what, iterations);
#else
  io[get_global_id(0)] = -1;
#endif
}

KERNEL(64) testThroughput(int what, int iterations, __global i64 *io) {
#if HAS_ASM
  size_t gid = get_global_id(0);
  io[gid] = run_throughput_case(what, iterations);
#else
  io[get_global_id(0)] = -1;
#endif
}
