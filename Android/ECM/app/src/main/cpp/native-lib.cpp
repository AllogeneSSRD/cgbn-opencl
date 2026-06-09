#include <jni.h>

#include <android/asset_manager_jni.h>

#include <string>

#include "kernel_assets.h"
#include "opencl_runtime.h"

static std::string prepend_opencl_error(JNIEnv* env, jstring j_err, const std::string& body) {
    std::string report;
    if (j_err != nullptr) {
        const char* err = env->GetStringUTFChars(j_err, nullptr);
        if (err != nullptr && err[0] != '\0') {
            report += "System.loadLibrary(OpenCL) failed:\n";
            report += err;
            report += "\n\n";
            env->ReleaseStringUTFChars(j_err, err);
        }
    }
    report += body;
    return report;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeProbe(JNIEnv* env, jobject /* this */, jstring j_opencl_load_error) {
    return env->NewStringUTF(prepend_opencl_error(env, j_opencl_load_error, probe_opencl()).c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeShortTest(JNIEnv* env, jobject /* this */) {
    return env->NewStringUTF(run_short_test().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ecm_MainActivity_nativeInitAssets(JNIEnv* env, jobject /* this */, jobject asset_manager) {
    set_kernel_asset_manager(AAssetManager_fromJava(env, asset_manager));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeAddSubBench(
        JNIEnv* env,
        jobject /* this */,
        jint bits,
        jint kernel_iters,
        jint instances,
        jint launch_repeats) {
    return env->NewStringUTF(
        run_addsub_bench(bits, kernel_iters, instances, launch_repeats).c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeBitBench(
        JNIEnv* env,
        jobject /* this */,
        jint limb_bits,
        jint elements,
        jint kernel_iters,
        jint launch_repeats) {
  return env->NewStringUTF(
      run_bit_bench(limb_bits, elements, kernel_iters, launch_repeats).c_str());
}
