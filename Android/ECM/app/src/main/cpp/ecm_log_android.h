#pragma once

#include <jni.h>

#include <string>

void android_ecm_log_begin(std::string* out);
void android_ecm_log_end();

/** Optional live log sink for ECM runs (JNI thread only). Pass null listener to disable. */
void android_ecm_log_set_listener(JNIEnv* env, jobject listener);
void android_ecm_log_clear_listener(JNIEnv* env);
bool android_ecm_log_listener_active();
