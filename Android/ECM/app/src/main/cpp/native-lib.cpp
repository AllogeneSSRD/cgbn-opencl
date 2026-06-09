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

extern "C" void ecm_read_gpu_stats_ex(int* out_busy_percent, long long* out_freq_hz, int* out_busy_src,
                                      int* out_freq_src, long long* out_max_freq_hz);

extern "C" JNIEXPORT void JNICALL
Java_com_example_ecm_MainActivity_nativeInitAssets(JNIEnv* env, jobject /* this */, jobject asset_manager) {
    set_kernel_asset_manager(AAssetManager_fromJava(env, asset_manager));
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_example_ecm_DevicePerfMonitor_nativeGpuStats(JNIEnv* env, jclass /* clazz */) {
    int busy = -1;
    long long freq_hz = -1;
    int busy_src = 0;
    int freq_src = 0;
    long long max_freq_hz = -1;
    ecm_read_gpu_stats_ex(&busy, &freq_hz, &busy_src, &freq_src, &max_freq_hz);
    jlong out[5] = {static_cast<jlong>(busy), static_cast<jlong>(freq_hz),
                    static_cast<jlong>(busy_src), static_cast<jlong>(freq_src),
                    static_cast<jlong>(max_freq_hz)};
    jlongArray arr = env->NewLongArray(5);
    if (arr != nullptr) {
        env->SetLongArrayRegion(arr, 0, 5, out);
    }
    return arr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeAddSubBench(
        JNIEnv* env,
        jobject /* this */,
        jint bits,
        jint kernel_iters,
        jint instances,
        jint launch_repeats,
        jint limb_bits) {
    return env->NewStringUTF(
        run_addsub_bench(bits, kernel_iters, instances, launch_repeats, limb_bits).c_str());
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
