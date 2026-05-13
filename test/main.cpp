#include <cstdio>
#include "bench_cgbn_2048.h"

int main() {
    const int iterations = 1e6;
    const int blocks = 96;
    const int threads = 256;

    bench_cgbn_2048_wapper(
        iterations,
        blocks,
        threads
    );

    return 0;
}