// Montgomery mul CIOS — AMDGCN inline asm helpers (512-bit bench).

#if defined(MONT_MUL_ASM_ENABLE) && defined(__AMDGCN__)

#ifndef MONT_OPT2_FIXED_LIMBS
#define MONT_OPT2_FIXED_LIMBS 16u
#endif

// v_mul for 32x32->64, then ulong add (matches CIOS C semantics mod 2^64).
static inline ulong asm_mac_row_step(ulong carry, uint tj, uint a, uint b, uint *out_tj) {
    uint plo, phi;
    __asm volatile(
        "v_mul_lo_u32 %[plo], %[a], %[b]\n\t"
        "v_mul_hi_u32 %[phi], %[a], %[b]"
        : [plo] "=&v"(plo), [phi] "=&v"(phi)
        : [a] "v"(a), [b] "v"(b));
    ulong uv = (ulong)tj + (ulong)plo + ((ulong)phi << 32) + carry;
    *out_tj = (uint)uv;
    return uv >> 32;
}

static inline ulong asm_red_step0_carry(uint t0, uint m, uint n0) {
    uint plo, phi;
    __asm volatile(
        "v_mul_lo_u32 %[plo], %[m], %[n0]\n\t"
        "v_mul_hi_u32 %[phi], %[m], %[n0]"
        : [plo] "=&v"(plo), [phi] "=&v"(phi)
        : [m] "v"(m), [n0] "v"(n0));
    ulong uv = (ulong)t0 + (ulong)plo + ((ulong)phi << 32);
    return uv >> 32;
}

#endif // MONT_MUL_ASM_ENABLE && __AMDGCN__
