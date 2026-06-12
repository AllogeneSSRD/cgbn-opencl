#include "kernel_assets.h"

#include <android/log.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

AAssetManager* g_assets = nullptr;

constexpr const char* kKernelAssetRoot = "cgbn/backends/opencl/kernels/";
constexpr const char* kEcmStage1AssetRoot = "kernels/opencl/";

void push_unique(std::vector<std::string>& paths, std::string path) {
    if (path.empty()) {
        return;
    }
    if (std::find(paths.begin(), paths.end(), path) == paths.end()) {
        paths.push_back(std::move(path));
    }
}

std::vector<std::string> kernel_asset_candidates(const char* rel_path) {
    std::vector<std::string> candidates;
    if (rel_path == nullptr || rel_path[0] == '\0') {
        return candidates;
    }

    std::string rel(rel_path);
    std::string tail = rel;
    if (rel.rfind(kEcmStage1AssetRoot, 0) == 0) {
        push_unique(candidates, rel);
        tail = rel.substr(std::strlen(kEcmStage1AssetRoot));
    } else if (rel.rfind(kKernelAssetRoot, 0) == 0) {
        tail = rel.substr(std::strlen(kKernelAssetRoot));
        push_unique(candidates, std::string("kernels/") + rel);
    } else {
        push_unique(candidates, std::string(kEcmStage1AssetRoot) + rel);
    }

    push_unique(candidates, std::string(kEcmStage1AssetRoot) + tail);
    push_unique(candidates, std::string("kernels/") + kKernelAssetRoot + tail);
    push_unique(candidates, std::string("kernels/") + tail);

    const size_t slash = tail.find_last_of('/');
    if (slash != std::string::npos) {
        push_unique(candidates, std::string("kernels/") + tail.substr(slash + 1));
    }

    return candidates;
}

bool read_asset_at(const std::string& asset_path, std::string& out) {
    AAsset* asset = AAssetManager_open(g_assets, asset_path.c_str(), AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        return false;
    }
    const off_t len = AAsset_getLength(asset);
    if (len <= 0) {
        AAsset_close(asset);
        return false;
    }
    std::vector<char> buf(static_cast<size_t>(len));
    const int read = AAsset_read(asset, buf.data(), static_cast<size_t>(len));
    AAsset_close(asset);
    if (read != len) {
        return false;
    }
    out.assign(buf.data(), static_cast<size_t>(len));
    return true;
}

} // namespace

void set_kernel_asset_manager(AAssetManager* mgr) {
    g_assets = mgr;
}

std::string load_kernel_asset(const char* rel_path) {
    if (g_assets == nullptr) {
        return {};
    }

    const std::vector<std::string> candidates = kernel_asset_candidates(rel_path);
    std::string data;
    for (const std::string& path : candidates) {
        if (read_asset_at(path, data)) {
            return data;
        }
    }

    std::string tried;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i > 0) {
            tried += ", ";
        }
        tried += candidates[i];
    }
    __android_log_print(ANDROID_LOG_ERROR, "ECM-OpenCL", "missing asset for %s (tried: %s)",
                        rel_path == nullptr ? "(null)" : rel_path, tried.c_str());
    return {};
}

namespace cgbn {
namespace opencl {

std::string android_load_kernel_asset(const char* rel_path) {
    return ::load_kernel_asset(rel_path);
}

} // namespace opencl
} // namespace cgbn
