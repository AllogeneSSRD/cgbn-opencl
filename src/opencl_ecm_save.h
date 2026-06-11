#pragma once

#include <gmp.h>

#include <cstdint>
#include <string>

/** Resolve relative paths against {@link opencl_ecm_set_work_dir}; absolute paths unchanged. */
std::string opencl_ecm_resolve_data_path(const char *path);

bool opencl_ecm_check_save_file_writable(const std::string &savefilename, bool saveappend);

std::string opencl_ecm_build_saved_n_expr(const std::string &original_expr, const mpz_t N,
                                          uint32_t curves, mpz_t *factors, int *array_found);

bool opencl_ecm_append_save_lines(const std::string &savefilename, const mpz_t N, double B1,
                                  uint32_t firstsigma, uint32_t curves, mpz_t *factors,
                                  const std::string &n_expr_save);
