// OpenCL kernels for ECM-style add/sub/mod operators — hot-loop benchmark edition.
// Each __kernel loads global→private once, loops inner_iters times (re-feeding
// output as new input), then stores private→global once.  Eliminates the
// global-memory-copy bottleneck that dominated the single-operation kernels.
//
// Kernel signatures take `uint inner_iters` as the LAST kernel argument.
// Callers pass 1u for single-operation (verification) or kernel_iterations
// for throughput measurement.

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

inline uint mp_add_n(uint *r, const uint *a, const uint *b, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}

inline void mp_sub_n(uint *r, const uint *a, const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i], nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

inline uint mp_sub_n_borrow(uint *r, const uint *a, const uint *b, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i], bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    return (uint)borrow;
}

inline int mp_ge(const uint *a, const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

#ifndef MP_ADD_MOD_FUSED_UNROLL
#define MP_ADD_MOD_FUSED_UNROLL 2
#endif

inline void mp_add_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong carry_add = 0ul, carry_sub = 1ul;
#if MP_ADD_MOD_FUSED_UNROLL == 2
    uint j = 0u;
    for (; j + 1u < limbs; j += 2u) {
        ulong s0 = (ulong)a[j] + (ulong)b[j] + carry_add;   carry_add = s0 >> 32;
        ulong t0 = (ulong)(uint)s0 + (ulong)(~N[j]) + carry_sub; carry_sub = t0 >> 32;
        r[j] = (uint)t0;
        ulong s1 = (ulong)a[j+1u] + (ulong)b[j+1u] + carry_add; carry_add = s1 >> 32;
        ulong t1 = (ulong)(uint)s1 + (ulong)(~N[j+1u]) + carry_sub; carry_sub = t1 >> 32;
        r[j+1u] = (uint)t1;
    }
    if (limbs & 1u) {
        ulong s = (ulong)a[j] + (ulong)b[j] + carry_add; carry_add = s>>32;
        ulong t = (ulong)(uint)s + (ulong)(~N[j]) + carry_sub; carry_sub = t>>32;
        r[j] = (uint)t;
    }
#else
    for (uint i = 0u; i < limbs; ++i) {
        ulong s = (ulong)a[i] + (ulong)b[i] + carry_add; carry_add = s>>32;
        ulong t = (ulong)(uint)s + (ulong)(~N[i]) + carry_sub; carry_sub = t>>32;
        r[i] = (uint)t;
    }
#endif
    if ((carry_add | carry_sub) != 0ul) return;
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) { ulong s = (ulong)r[i] + (ulong)N[i] + c; r[i] = (uint)s; c = s>>32; }
}

inline int mp_sub_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i], bv = (ulong)b[i], w = av - bv - borrow;
        r[i] = (uint)w; borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    if (borrow) { (void)mp_add_n(r, r, N, limbs); return 1; }
    return 0;
}

inline void mp_add_mod_legacy(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) { ulong s = (ulong)a[i] + (ulong)b[i] + carry; r[i] = (uint)s; carry = s>>32; }
    if (carry != 0ul || mp_ge(r, N, limbs)) mp_sub_n(r, r, N, limbs);
}

inline void mp_add_mod_mask(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    uint S[MAX_LIMBS]; ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) { ulong s = (ulong)a[i] + (ulong)b[i] + carry; S[i] = (uint)s; carry = s>>32; }
    uint borrow = mp_sub_n_borrow(r, S, N, limbs);
    uint need_sub = (uint)(carry | (borrow == 0u)), mask = 0u - need_sub;
    for (uint i = 0u; i < limbs; ++i) r[i] = (r[i] & mask) | (S[i] & ~mask);
}

// ═══ Hot-loop bench kernels — load once, loop iters, store once ═══════════

__kernel void ecm_mp_add_n(__global const uint *a,__global const uint *b,__global uint *out,uint limbs,uint inner_iters){
    uint gid=get_global_id(0),base=gid*limbs;
    uint x[MAX_LIMBS],y[MAX_LIMBS],r[MAX_LIMBS];
    for(uint i=0;i<limbs;++i){x[i]=a[base+i];y[i]=b[base+i];}
    for(uint k=0;k<inner_iters;++k){
        (void)mp_add_n(r,(k==0?x:r),y,limbs);
        if(k+1<inner_iters) for(uint i=0;i<limbs;++i)x[i]=r[i];
    }
    for(uint i=0;i<limbs;++i)out[base+i]=r[i];
}

__kernel void ecm_mp_sub_n(__global const uint *a,__global const uint *n,__global uint *out,uint limbs,uint inner_iters){
    uint gid=get_global_id(0),base=gid*limbs;
    uint x[MAX_LIMBS],m[MAX_LIMBS],r[MAX_LIMBS];
    for(uint i=0;i<limbs;++i){x[i]=a[base+i];m[i]=n[base+i];}
    mp_sub_n(r,x,m,limbs);
    for(uint k=1;k<inner_iters;++k){mp_sub_n(r,(k%2?r:x),m,limbs);}
    for(uint i=0;i<limbs;++i)out[base+i]=r[i];
}

__kernel void ecm_mp_add_mod_legacy(__global const uint *a,__global const uint *b,__global const uint *n,__global uint *out,uint limbs,uint inner_iters){
    uint gid=get_global_id(0),base=gid*limbs;
    uint x[MAX_LIMBS],y[MAX_LIMBS],m[MAX_LIMBS],r[MAX_LIMBS];
    for(uint i=0;i<limbs;++i){x[i]=a[base+i];y[i]=b[base+i];m[i]=n[base+i];}
    for(uint k=0;k<inner_iters;++k){
        mp_add_mod_legacy(r,(k==0?x:r),y,m,limbs);
        if(k+1<inner_iters) for(uint i=0;i<limbs;++i)x[i]=r[i];
    }
    for(uint i=0;i<limbs;++i)out[base+i]=r[i];
}

__kernel void ecm_mp_add_mod_mask(__global const uint *a,__global const uint *b,__global const uint *n,__global uint *out,uint limbs,uint inner_iters){
    uint gid=get_global_id(0),base=gid*limbs;
    uint x[MAX_LIMBS],y[MAX_LIMBS],m[MAX_LIMBS],r[MAX_LIMBS];
    for(uint i=0;i<limbs;++i){x[i]=a[base+i];y[i]=b[base+i];m[i]=n[base+i];}
    for(uint k=0;k<inner_iters;++k){
        mp_add_mod_mask(r,(k==0?x:r),y,m,limbs);
        if(k+1<inner_iters) for(uint i=0;i<limbs;++i)x[i]=r[i];
    }
    for(uint i=0;i<limbs;++i)out[base+i]=r[i];
}

__kernel void ecm_mp_add_mod_fused(__global const uint *a,__global const uint *b,__global const uint *n,__global uint *out,uint limbs,uint inner_iters){
    uint gid=get_global_id(0),base=gid*limbs;
    uint x[MAX_LIMBS],y[MAX_LIMBS],m[MAX_LIMBS],r[MAX_LIMBS];
    for(uint i=0;i<limbs;++i){x[i]=a[base+i];y[i]=b[base+i];m[i]=n[base+i];}
    for(uint k=0;k<inner_iters;++k){
        mp_add_mod(r,(k==0?x:r),y,m,limbs);
        if(k+1<inner_iters) for(uint i=0;i<limbs;++i)x[i]=r[i];
    }
    for(uint i=0;i<limbs;++i)out[base+i]=r[i];
}

// Backward-compat: kept for existing hot-unroll path, identical to fused + inner_iters
__kernel void ecm_mp_add_mod_fused_hot(__global const uint *a,__global const uint *b,__global const uint *n,__global uint *out,uint limbs,uint inner_iters){
    uint gid=get_global_id(0),base=gid*limbs;
    uint x[MAX_LIMBS],y[MAX_LIMBS],m[MAX_LIMBS],r[MAX_LIMBS];
    for(uint i=0;i<limbs;++i){x[i]=a[base+i];y[i]=b[base+i];m[i]=n[base+i];}
    for(uint k=0;k<inner_iters;++k){
        mp_add_mod(r,(k==0?x:r),y,m,limbs);
        if(k+1<inner_iters) for(uint i=0;i<limbs;++i)x[i]=r[i];
        y[0]=y[0]+1u;
    }
    for(uint i=0;i<limbs;++i)out[base+i]=r[i];
}

__kernel void ecm_mp_sub_mod(__global const uint *a,__global const uint *b,__global const uint *n,__global uint *out,uint limbs,uint inner_iters){
    uint gid=get_global_id(0),base=gid*limbs;
    uint x[MAX_LIMBS],y[MAX_LIMBS],m[MAX_LIMBS],r[MAX_LIMBS];
    for(uint i=0;i<limbs;++i){x[i]=a[base+i];y[i]=b[base+i];m[i]=n[base+i];}
    for(uint k=0;k<inner_iters;++k){
        (void)mp_sub_mod(r,(k==0?x:r),y,m,limbs);
        if(k+1<inner_iters) for(uint i=0;i<limbs;++i)x[i]=r[i];
    }
    for(uint i=0;i<limbs;++i)out[base+i]=r[i];
}
