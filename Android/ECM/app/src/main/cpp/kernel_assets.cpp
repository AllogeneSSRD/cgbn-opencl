#include "kernel_assets.h"

#include <android/log.h>

#include <vector>

namespace {

AAssetManager* g_assets = nullptr;

} // namespace

void set_kernel_asset_manager(AAssetManager* mgr) {
    g_assets = mgr;
}

std::string load_kernel_asset(const char* rel_path) {
    if (g_assets == nullptr || rel_path == nullptr || rel_path[0] == '\0') {
        return {};
    }
    std::string asset_path = std::string("kernels/") + rel_path;
    AAsset* asset = AAssetManager_open(g_assets, asset_path.c_str(), AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ECM-OpenCL", "missing asset: %s", asset_path.c_str());
        return {};
    }
    const off_t len = AAsset_getLength(asset);
    if (len <= 0) {
        AAsset_close(asset);
        return {};
    }
    std::vector<char> buf(static_cast<size_t>(len));
    const int read = AAsset_read(asset, buf.data(), static_cast<size_t>(len));
    AAsset_close(asset);
    if (read != len) {
        return {};
    }
    return std::string(buf.data(), buf.size());
}
