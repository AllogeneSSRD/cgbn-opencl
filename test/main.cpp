#include <cstdio>
#include "bench_cgbn_2048.h"

int main() {
    const int iterations = 100000;
    const int blocks = 24;
    const int threads = 128;

    bench_cgbn_2048_wapper(
        iterations,
        blocks,
        threads
    );

    return 0;
}