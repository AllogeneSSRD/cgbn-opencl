#include <jni.h>

#include <string>

#include "opencl_probe.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */,
        jstring j_opencl_load_error) {
    std::string report;
    if (j_opencl_load_error != nullptr) {
        const char* err = env->GetStringUTFChars(j_opencl_load_error, nullptr);
        if (err != nullptr && err[0] != '\0') {
            report += "System.loadLibrary(OpenCL) failed:\n";
            report += err;
            report += "\n\n";
            env->ReleaseStringUTFChars(j_opencl_load_error, err);
        }
    }
    report += probe_opencl();
    return env->NewStringUTF(report.c_str());
}
