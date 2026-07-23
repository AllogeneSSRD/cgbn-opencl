// Sliced CIOS 8192 �� LDS cond-sub, lane 0 serial, scatter to global directly.
#define DLIMBS 8u
#define LIMBS  (32u*DLIMBS)
#define D2     (DLIMBS+2u)

__kernel __attribute__((reqd_work_group_size(32u, 1u, 1u)))
void sliced_cios_mul_8192(__global const uint *A, __global const uint *B, __global const uint *N, __global uint *R, uint np0){
    uint lid = get_local_id(0);
    __local uint L_T[32u * D2 + DLIMBS];
    __local uint L_N[LIMBS];
    for (uint i = lid; i < LIMBS; i += 32u) L_N[i] = N[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint my_T[DLIMBS], my_A[DLIMBS], my_B[DLIMBS], my_N[DLIMBS];
    for (uint d = 0u; d < DLIMBS; ++d) {
        my_T[d] = 0u; my_A[d] = A[lid*DLIMBS+d]; my_B[d] = B[lid*DLIMBS+d]; my_N[d] = N[lid*DLIMBS+d];
    }
    uint tN = 0u, tN1 = 0u;

    for (uint i = 0u; i < LIMBS; ++i) {
        uint i_lane = i / DLIMBS, i_off = i % DLIMBS;
        uint A_i = __builtin_amdgcn_ds_bpermute(i_lane * 4u, my_A[i_off]);
        uint carry = 0u;
        for (uint d = 0u; d < DLIMBS; ++d) {ulong uv = (ulong)my_T[d] + (ulong)A_i * (ulong)my_B[d] + carry; my_T[d] = (uint)uv; carry = (uint)(uv >> 32);}
        uint ci = (lid == 0u) ? 0u : __builtin_amdgcn_ds_bpermute(((lid - 1u) & 31u) * 4u, carry);
        for (uint d = 0u; d < DLIMBS && ci; ++d) {ulong uv = (ulong)my_T[d] + (ulong)ci; my_T[d] = (uint)uv; ci = (uint)(uv >> 32);}
        uint C31 = __builtin_amdgcn_ds_bpermute(31u * 4u, carry);
        {ulong t = (ulong)tN + (ulong)C31; tN = (uint)t; tN1 += (uint)(t >> 32);}

        uint m = 0u;if(lid==0u)m=(uint)((ulong)my_T[0]*(ulong)np0);m=__builtin_amdgcn_readfirstlane(m);
        carry = 0u;
        for (uint d = 0u; d < DLIMBS; ++d) {ulong uv = (ulong)my_T[d] + (ulong)m * (ulong)my_N[d] + carry; my_T[d] = (uint)uv; carry = (uint)(uv >> 32);}
        ci = (lid == 0u) ? 0u : __builtin_amdgcn_ds_bpermute(((lid - 1u) & 31u) * 4u, carry);
        for (uint d = 0u; d < DLIMBS && ci; ++d) {ulong uv = (ulong)my_T[d] + (ulong)ci; my_T[d] = (uint)uv; ci = (uint)(uv >> 32);}
        C31 = __builtin_amdgcn_ds_bpermute(31u * 4u, carry);
        {ulong t = (ulong)tN + (ulong)C31; tN = (uint)t; tN1 += (uint)(t >> 32);}

        for (uint d = 0u; d < DLIMBS; ++d) L_T[lid * D2 + d] = my_T[d];
        if (lid == 31u) {L_T[31u * D2 + DLIMBS] = tN; L_T[31u * D2 + DLIMBS + 1u] = tN1;}
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < 31u) {for(uint d=0u;d<DLIMBS;++d)my_T[d]=L_T[(lid+1u)*D2+d];}
        else {my_T[0]=L_T[31u*D2+DLIMBS];for(uint d=1u;d<DLIMBS;++d)my_T[d]=L_T[31u*D2+d-1u];}
        tN=L_T[31u*D2+DLIMBS+1u];tN1=0u;
    }

    for (uint d = 0u; d < DLIMBS; ++d) L_T[lid*D2+d] = my_T[d];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        uint borrow=0u,D[DLIMBS];
        for(uint k=0u;k<LIMBS;++k){
            uint kl=k/DLIMBS,ko=k%DLIMBS;
            uint tv=L_T[kl*D2+ko],nv=L_N[k];
            ulong w=(ulong)tv-(ulong)nv-(ulong)borrow;
            if(kl==0u)D[ko]=(uint)w;
            borrow=(tv<nv+borrow)?1u:0u;
        }
        uint ns=(borrow==0u)?1u:0u;uint mk=0u-ns;
        for(uint k=0u;k<LIMBS;++k){
            uint kl=k/DLIMBS,ko=k%DLIMBS;
            uint tv=L_T[kl*D2+ko];
            R[k] = (mk!=0u) ? (uint)((ulong)tv-(ulong)L_N[k]) : tv;
        }
    }
}
