#pragma once

#include <android/asset_manager.h>
#include <string>

void set_kernel_asset_manager(AAssetManager* mgr);
std::string load_kernel_asset(const char* rel_path);
