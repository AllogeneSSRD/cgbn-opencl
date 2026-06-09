#include "jni_utf8.h"

std::string sanitize_modified_utf8(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size();) {
        const unsigned char c = static_cast<unsigned char>(input[i]);
        if (c == 0) {
            out += '?';
            ++i;
            continue;
        }
        if (c < 0x80) {
            out += static_cast<char>(c);
            ++i;
            continue;
        }

        size_t len = 0;
        if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        } else {
            out += '?';
            ++i;
            continue;
        }

        if (i + len > input.size()) {
            out += '?';
            ++i;
            continue;
        }

        bool ok = true;
        for (size_t j = 1; j < len; ++j) {
            const unsigned char cc = static_cast<unsigned char>(input[i + j]);
            if ((cc & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            out += '?';
            ++i;
            continue;
        }

        out.append(input, i, len);
        i += len;
    }
    return out;
}

jstring new_jstring_utf8(JNIEnv* env, const std::string& text) {
    if (env == nullptr) {
        return nullptr;
    }
    const std::string safe = sanitize_modified_utf8(text);
    jstring result = env->NewStringUTF(safe.c_str());
    if (result != nullptr) {
        return result;
    }
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
    return env->NewStringUTF("(native output contained invalid UTF-8; sanitized copy failed)");
}
