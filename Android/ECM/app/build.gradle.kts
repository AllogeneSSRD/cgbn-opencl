plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.ecm"
    compileSdk {
        version = release(37) {
//            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.ecm"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
    }
    packaging {
        jniLibs {
            // Never ship vendor-pulled OpenCL (wrong ELF page alignment on 16 KB devices).
            excludes += setOf(
                "**/libOpenCL.so",
                "**/libcutils.so",
                "**/libvndksupport.so",
            )
        }
    }
}

dependencies {
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}