#include <jni.h>

#include <string>

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
