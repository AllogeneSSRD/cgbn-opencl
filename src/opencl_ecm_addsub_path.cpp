#include "opencl_ecm_addsub_path.h"

#include "opencl_ecm_path_registry.h"

#include <cstring>

namespace {

bool aliases_contain(const char *const *aliases, const char *path) {
    if (aliases == nullptr || path == nullptr) {
        return false;
    }
    for (const char *const *p = aliases; *p != nullptr; ++p) {
        if (strcmp(path, *p) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

int opencl_ecm_parse_addsub_path(const char *path) {
    if (opencl_ecm_path_is_auto(path)) {
        return -1;
    }
    for (size_t i = 0; i < opencl_ecm_addmod_registry_count(); ++i) {
        const EcmAddModPathDescriptor *desc = opencl_ecm_addmod_registry_entry(i);
        if (desc != nullptr && aliases_contain(desc->aliases, path)) {
            return desc->path_id;
        }
    }
    for (size_t i = 0; i < opencl_ecm_submod_registry_count(); ++i) {
        const EcmSubModPathDescriptor *desc = opencl_ecm_submod_registry_entry(i);
        if (desc != nullptr && aliases_contain(desc->aliases, path)) {
            return desc->path_id;
        }
    }
    return -2;
}

const char *opencl_ecm_addsub_path_name(int path_id) {
    const EcmAddModPathDescriptor *add_d = opencl_ecm_addmod_path_descriptor(path_id);
    if (add_d != nullptr && add_d->cl_name != nullptr) {
        return add_d->cl_name;
    }
    const EcmSubModPathDescriptor *sub_d = opencl_ecm_submod_path_descriptor(path_id);
    if (sub_d != nullptr && sub_d->cl_name != nullptr) {
        return sub_d->cl_name;
    }
    return "unknown";
}

bool opencl_ecm_addsub_path_needs_asm_b32(int path_id) {
    const EcmAddModPathDescriptor *add_d = opencl_ecm_addmod_path_descriptor(path_id);
    if (add_d != nullptr && add_d->needs_asm_b32) {
        return true;
    }
    const EcmSubModPathDescriptor *sub_d = opencl_ecm_submod_path_descriptor(path_id);
    return sub_d != nullptr && sub_d->needs_asm_b32;
}

bool opencl_ecm_addsub_path_needs_asm_b16(int path_id) {
    const EcmAddModPathDescriptor *add_d = opencl_ecm_addmod_path_descriptor(path_id);
    if (add_d != nullptr && add_d->needs_asm_b16) {
        return true;
    }
    const EcmSubModPathDescriptor *sub_d = opencl_ecm_submod_path_descriptor(path_id);
    return sub_d != nullptr && sub_d->needs_asm_b16;
}

const EcmAddModPathDescriptor *opencl_ecm_resolve_addsub_add_path(const char *path, uint32_t limbs,
                                                                  bool is_amd) {
    return opencl_ecm_resolve_addmod_path(path, limbs, is_amd);
}

const EcmSubModPathDescriptor *opencl_ecm_resolve_addsub_sub_path(const char *path, uint32_t limbs,
                                                                  bool is_amd) {
    return opencl_ecm_resolve_submod_path(path, limbs, is_amd);
}

void opencl_ecm_print_available_kernels(FILE *out) {
    if (out == nullptr) {
        out = stdout;
    }
    fprintf(out, "ECM OpenCL kernels: mul/sqr and add/sub paths are resolved independently.\n");
    fprintf(out, "See docs/DEV_OPERATOR_PATH_REGISTRY.md\n");
}
