#include "opencl_ecm_checkpoint.h"

#include "opencl_ecm_log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

const char *opencl_ecm_checkpoint_filename(const mpz_t N) {
    static char filename[512];
    const size_t nbits = mpz_sizeinbase(N, 2);
    char *N_str = mpz_get_str(nullptr, 16, N);
    if (!N_str) {
        snprintf(filename, sizeof(filename), ".ecm_ckpt_%zu_alloc_fail.dat", nbits);
        return filename;
    }
    const size_t len = strlen(N_str);
    char first_hex[16] = {0};
    char last_hex[16] = {0};
    strncpy(first_hex, N_str, (len >= 8u) ? 8u : len);
    if (len > 8u) {
        strncpy(last_hex, N_str + len - 8u, 8u);
    }
    if (len > 8u && last_hex[0] != '\0') {
        snprintf(filename, sizeof(filename), ".ecm_ckpt_%zu_%s_%s.dat", nbits, first_hex, last_hex);
    } else {
        snprintf(filename, sizeof(filename), ".ecm_ckpt_%zu_%s.dat", nbits, first_hex);
    }
    free(N_str);
    return filename;
}

int opencl_ecm_checkpoint_save(const char *filename, const opencl_ecm_checkpoint_header_t *header,
                               const uint32_t *data, size_t data_size) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        ecm_ts_fprintf(stderr, "Warning: Could not open checkpoint file '%s' for writing\n",
                       filename);
        return ECM_ERROR;
    }
    if (fwrite(header, sizeof(opencl_ecm_checkpoint_header_t), 1, f) != 1) {
        ecm_ts_fprintf(stderr, "Error writing checkpoint header\n");
        fclose(f);
        return ECM_ERROR;
    }
    if (fwrite(data, 1, data_size, f) != data_size) {
        ecm_ts_fprintf(stderr, "Error writing checkpoint data\n");
        fclose(f);
        return ECM_ERROR;
    }
    fclose(f);
    ecm_ts_fprintf(stderr, "Checkpoint saved: s_partial=%llu/%llu (%.1f%%)\n",
                   (unsigned long long)header->s_partial,
                   (unsigned long long)header->s_num_bits,
                   header->s_num_bits ? (100.0 * header->s_partial / header->s_num_bits) : 0.0);
    return ECM_NO_FACTOR_FOUND;
}

int opencl_ecm_checkpoint_load(const char *filename, opencl_ecm_checkpoint_header_t *header,
                               uint32_t **data_ptr, size_t *data_size_ptr) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        return -1;
    }
    opencl_ecm_checkpoint_header_t temp_header{};
    if (fread(&temp_header, sizeof(opencl_ecm_checkpoint_header_t), 1, f) != 1) {
        ecm_ts_fprintf(stderr, "Warning: Could not read checkpoint header\n");
        fclose(f);
        return -1;
    }
    if (temp_header.magic != OPENCL_ECM_CHECKPOINT_MAGIC) {
        ecm_ts_fprintf(stderr, "Warning: Checkpoint file has invalid magic number\n");
        fclose(f);
        return -1;
    }
    if (temp_header.version != OPENCL_ECM_CHECKPOINT_VERSION) {
        ecm_ts_fprintf(stderr, "Warning: Checkpoint version mismatch (expected %d, got %u)\n",
                       OPENCL_ECM_CHECKPOINT_VERSION, temp_header.version);
        fclose(f);
        return -1;
    }
    if (temp_header.data_size == 0u) {
        ecm_ts_fprintf(stderr, "Warning: Checkpoint data_size is zero\n");
        fclose(f);
        return -1;
    }
    uint32_t *data = (uint32_t *)malloc((size_t)temp_header.data_size);
    if (!data) {
        ecm_ts_fprintf(stderr, "Error: Could not allocate memory for checkpoint data\n");
        fclose(f);
        return -1;
    }
    if (fread(data, 1, (size_t)temp_header.data_size, f) != (size_t)temp_header.data_size) {
        ecm_ts_fprintf(stderr, "Warning: Could not read checkpoint data completely\n");
        free(data);
        fclose(f);
        return -1;
    }
    fclose(f);
    *header = temp_header;
    *data_ptr = data;
    *data_size_ptr = (size_t)temp_header.data_size;
    const time_t now = time(nullptr);
    const time_t age = now - (time_t)temp_header.timestamp;
    ecm_ts_fprintf(stderr,
                   "Checkpoint loaded: s_partial=%llu/%llu (%.1f%%), age=%lld seconds\n",
                   (unsigned long long)temp_header.s_partial,
                   (unsigned long long)temp_header.s_num_bits,
                   temp_header.s_num_bits ? (100.0 * temp_header.s_partial / temp_header.s_num_bits)
                                          : 0.0,
                   (long long)age);
    return ECM_NO_FACTOR_FOUND;
}

int opencl_ecm_checkpoint_remove(const char *filename) {
    return remove(filename);
}
