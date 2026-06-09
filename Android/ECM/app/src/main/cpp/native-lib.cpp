#include <jni.h>

#include <string>

#include "opencl_probe.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_stringFromJNI(JNIEnv* env, jobject /* this */) {
    const std::string report = probe_opencl();
    return env->NewStringUTF(report.c_str());
}
