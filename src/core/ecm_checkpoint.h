#pragma once

#include "ecm.h"

#include <gmp.h>

#include <cstddef>
#include <cstdint>
#include <string>

#define OPENCL_ECM_CHECKPOINT_MAGIC 0x45555047u  // "GPUE" little-endian
#define OPENCL_ECM_CHECKPOINT_VERSION 3

struct opencl_ecm_checkpoint_header_t {
    uint32_t magic;
    uint32_t version;
    uint64_t s_partial;
    uint64_t s_num_bits;
    int32_t batches_complete;
    uint32_t curves;
    uint32_t sigma;
    uint32_t BITS;
    uint32_t TPI;
    uint64_t data_size;
    int64_t timestamp;
};

static_assert(sizeof(opencl_ecm_checkpoint_header_t) == 64,
              "opencl_ecm_checkpoint_header_t layout must match cgbn_stage1.cu");

/** Writable base directory for checkpoint/save relative paths (e.g. Android app data root). */
void opencl_ecm_set_work_dir(const char *dir);
const char *opencl_ecm_get_work_dir();

/** Resolve relative paths against work_dir; absolute paths are unchanged. */
std::string opencl_ecm_resolve_data_path_buf(const char *path);

/** Create parent directories for a file path; returns false on failure. */
bool opencl_ecm_ensure_parent_dir(const char *filepath);

const char *opencl_ecm_checkpoint_filename(const mpz_t N);
int opencl_ecm_checkpoint_save(const char *filename, const opencl_ecm_checkpoint_header_t *header,
                               const uint32_t *data, size_t data_size);
int opencl_ecm_checkpoint_load(const char *filename, opencl_ecm_checkpoint_header_t *header,
                               uint32_t **data_ptr, size_t *data_size_ptr);
int opencl_ecm_checkpoint_remove(const char *filename);
