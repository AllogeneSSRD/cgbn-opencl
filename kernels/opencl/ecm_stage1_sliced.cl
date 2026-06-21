// LDS-gather-based sliced: no ds_bpermute, all standard operators, lane 0 only
__kernel __attribute__((reqd_work_group_size(32u,1u,1u)))
void kernel_double_add_sliced(__global const uint*sb,ulong sn,ulong st,ulong si,__global uint*d,uint cnt,uint sig,uint NP0,uint lim){
    if(lim!=32)return;uint lid=get_local_id(0),ins=get_group_id(0);if(ins>=cnt)return;
    uint bs=ins*160u;
    __local uint Ld[160]; // LDS for gather
    uint N_my=d[bs+lid],aX_my=d[bs+32+lid],aZ_my=d[bs+64+lid],bX_my=d[bs+96+lid],bZ_my=d[bs+128+lid];
    // Gather via LDS
    Ld[lid]=N_my; Ld[32+lid]=aX_my; Ld[64+lid]=aZ_my; Ld[96+lid]=bX_my; Ld[128+lid]=bZ_my;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid!=0u)return;
    uint N_full[32],aX_full[32],aZ_full[32],bX_full[32],bZ_full[32];
    for(uint i=0;i<32;i++){N_full[i]=Ld[i];aX_full[i]=Ld[32+i];aZ_full[i]=Ld[64+i];bX_full[i]=Ld[96+i];bZ_full[i]=Ld[128+i];}

    uint dv=sig+ins;int sw=0;ulong se=st+si;if(se>sn)se=sn;
    for(ulong b=st;b<se;b++){
        ulong nth=sn-1-b;uint li=(uint)(nth>>5),bi=(uint)(nth&31);int bit=(int)((sb[li]>>bi)&1u);
        if(bit!=sw){sw=!sw;for(uint i=0;i<32;i++){uint t=aX_full[i];aX_full[i]=bX_full[i];bX_full[i]=t;t=aZ_full[i];aZ_full[i]=bZ_full[i];bZ_full[i]=t;}}
        uint t[32],CB[32],DA[32],AA[32],BB[32],K[32],dK[32],qq[32],uu[32],ww[32],vv[32];
        for(uint i=0;i<32;i++){qq[i]=aX_full[i];uu[i]=aZ_full[i];ww[i]=bX_full[i];vv[i]=bZ_full[i];}
        add_mod_asm_1024b(t,vv,ww,N_full,32u);(void)sub_mod_asm_1024b(vv,vv,ww,N_full,32u);
        add_mod_asm_1024b(ww,uu,qq,N_full,32u);(void)sub_mod_asm_1024b(uu,uu,qq,N_full,32u);
        mont_mul_unroll_1024b(CB,t,uu,N_full,1u,32u);mont_mul_unroll_1024b(DA,vv,ww,N_full,1u,32u);
        mont_mul_unroll_1024b(AA,ww,ww,N_full,1u,32u);mont_mul_unroll_1024b(BB,uu,uu,N_full,1u,32u);
        mont_mul_unroll_1024b(qq,AA,BB,N_full,1u,32u);
        (void)sub_mod_asm_1024b(K,AA,BB,N_full,32u);for(uint i=0;i<32;i++)dK[i]=K[i];
        special_mult_ui32_unroll_1024b(dK,dv,N_full,1u,32u);
        add_mod_asm_1024b(uu,BB,dK,N_full,32u);mont_mul_unroll_1024b(uu,K,uu,N_full,1u,32u);
        add_mod_asm_1024b(ww,DA,CB,N_full,32u);(void)sub_mod_asm_1024b(vv,DA,CB,N_full,32u);
        mont_mul_unroll_1024b(ww,ww,ww,N_full,1u,32u);mont_mul_unroll_1024b(vv,vv,vv,N_full,1u,32u);
        mp_shift_left_1_mod(vv,vv,N_full,32u);
        for(uint i=0;i<32;i++){aX_full[i]=qq[i];aZ_full[i]=uu[i];bX_full[i]=ww[i];bZ_full[i]=vv[i];}
    }
    if(sw){for(uint i=0;i<32;i++){uint t=aX_full[i];aX_full[i]=bX_full[i];bX_full[i]=t;t=aZ_full[i];aZ_full[i]=bZ_full[i];bZ_full[i]=t;}}
    for(uint i=0;i<32;i++){d[bs+32+i]=aX_full[i];d[bs+64+i]=aZ_full[i];d[bs+96+i]=bX_full[i];d[bs+128+i]=bZ_full[i];}
}
