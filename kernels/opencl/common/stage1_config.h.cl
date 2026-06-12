// ECM stage1 shared compile-time configuration (prepended first).
// OpenCL ECM Stage 1 — Montgomery ladder (double_add_v2), ported from test/cgbn_stage1.cu

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

#ifndef TPI
#define TPI 8
#endif

#ifndef ECM_STAGE1_FORCE_NORMALIZE
#define ECM_STAGE1_FORCE_NORMALIZE 0
#endif

#ifndef ECM_STAGE1_MUL_PATH
#define ECM_STAGE1_MUL_PATH 0
#endif

#ifndef ECM_STAGE1_SQR_PATH
#define ECM_STAGE1_SQR_PATH 0
#endif

#ifndef ECM_STAGE1_COOP_WG
#define ECM_STAGE1_COOP_WG 0
#endif

#ifndef ECM_STAGE1_COOP_SCRATCH_U32
#define ECM_STAGE1_COOP_SCRATCH_U32 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_UNROLL32
#define ECM_STAGE1_MUL_FORCE_UNROLL32 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_UNROLL384
#define ECM_STAGE1_MUL_FORCE_UNROLL384 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_PRIV_OPT
#define ECM_STAGE1_MUL_FORCE_PRIV_OPT 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_UNROLL32
#define ECM_STAGE1_SQR_FORCE_UNROLL32 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_UNROLL384
#define ECM_STAGE1_SQR_FORCE_UNROLL384 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_PRIV_OPT
#define ECM_STAGE1_SQR_FORCE_PRIV_OPT 0
#endif

#ifndef ECM_STAGE1_384_LIMBS
#define ECM_STAGE1_384_LIMBS 12u
#endif

#ifndef ECM_STAGE1_512_CONTAINER_LIMBS
#define ECM_STAGE1_512_CONTAINER_LIMBS 16u
#endif

// Path ids: 0=unroll64_4096, 1=unroll64_4096_mt2, 2=fips4096, 3=fips4096_mt8, 4=fips4096_mt16
#if ECM_STAGE1_COOP_WG > 1
#define ECM_STAGE1_USE_COOP_WG 1
#else
#define ECM_STAGE1_USE_COOP_WG 0
#endif

#define MONT_FIXED_4096_LIMBS 128u
#define ECM_STAGE1_MT2_LOCAL_U32 (MONT_FIXED_4096_LIMBS + 2u + MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + 3u)
