#pragma once

#include <cstdint>
#include <string>

struct EcmAndroidRunRequest {
    std::string n_expr;
    double b1 = 2000.0;
    double b2 = 0.0;
    uint32_t gpu_curves = 64;
    int device_index = 0;
    bool verbose = false;
    double gpu_ckpt_sec = 600.0;
    bool sigma_fixed = false;
    uint32_t sigma = 0;
    std::string mul_path;
    std::string sqr_path;
    std::string add_path;
    std::string sub_path;
    std::string special_mult_path;
    std::string save_file;
    bool save_append = false;
};

std::string run_ecm_android(const EcmAndroidRunRequest& req);
