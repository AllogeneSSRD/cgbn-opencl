#ifndef _CGBN_STAGE1_H
#define _CGBN_STAGE1_H 1

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void bench_cgbn_2048_wapper(
    int iterations,
    int blocks,
    int threads);

#ifdef __cplusplus
}
#endif

#endif /* _CGBN_STAGE1_H */