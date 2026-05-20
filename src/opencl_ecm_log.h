#pragma once

#include <cstdarg>
#include <cstdio>

// Install timestamp prefixing for std::cout/std::cerr.
void ecm_install_timestamped_iostreams();
bool ecm_log_timestamp_enabled();

// Timestamped wrappers for C stdio output.
int ecm_ts_vfprintf(FILE *stream, const char *fmt, va_list ap);
int ecm_ts_fprintf(FILE *stream, const char *fmt, ...);
