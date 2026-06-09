#pragma once

#include <jni.h>

#include <string>

// Strip/replace bytes invalid for JNI NewStringUTF (Modified UTF-8).
std::string sanitize_modified_utf8(const std::string& input);

jstring new_jstring_utf8(JNIEnv* env, const std::string& text);
