#include <cstdio>
#include "bench_cgbn_4096.h"

int main() {
    const int iterations = 10000;
    const int blocks = 24;
    const int threads = 256;

    bench_cgbn_4096_wapper(
        iterations,
        blocks,
        threads
    );

    return 0;
}
