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
        minSdk = 31
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

val mpaRoot = rootProject.projectDir.parentFile.parentFile

tasks.register<Copy>("syncAddsubKernels") {
    from(mpaRoot.resolve("cgbn/backends/opencl/kernels")) {
        include("ecm_addsub_bench.cl")
        include("mp_addsub/generated/add_fused_unroll_manual.cl")
        include("mp_addsub/generated/sub_fused_unroll_manual.cl")
        include("mp_addsub/generated/fused_unroll_auto.cl")
    }
    into(layout.projectDirectory.dir("src/main/assets/kernels/cgbn/backends/opencl/kernels"))
}

tasks.named("preBuild") {
    dependsOn("syncAddsubKernels")
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