import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localPropertiesFile.inputStream().use { localProperties.load(it) }
}

/** CMake-friendly path (forward slashes). */
fun String.toCmakePath(): String = replace('\\', '/')

val ecmAndroidGmpRoot: String? = sequenceOf(
    localProperties.getProperty("ecm.android.gmp.root"),
    project.findProperty("ecm.android.gmp.root") as String?,
).mapNotNull { it?.trim()?.takeIf(String::isNotEmpty) }
    .firstOrNull()
    ?.toCmakePath()

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

        externalNativeBuild {
            cmake {
                if (ecmAndroidGmpRoot != null) {
                    arguments += listOf("-DECM_ANDROID_GMP_ROOT=$ecmAndroidGmpRoot")
                }
            }
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

val addsubKernelIncludes = arrayOf(
    "ecm_addsub_bench.cl",
    "mp_addsub/generated/add_fused_unroll_manual.cl",
    "mp_addsub/generated/sub_fused_unroll_manual.cl",
    "mp_addsub/generated/fused_unroll_auto.cl",
    "mp_addsub/limb24_addsub.cl",
)

tasks.register<Copy>("syncAddsubKernels") {
    from(mpaRoot.resolve("cgbn/backends/opencl/kernels")) {
        addsubKernelIncludes.forEach { include(it) }
    }
    into(layout.projectDirectory.dir("src/main/assets/kernels/cgbn/backends/opencl/kernels"))
}

// Flat mirror (kernels/<name>.cl) for legacy asset paths on device.
tasks.register<Copy>("syncAddsubKernelsFlat") {
    from(mpaRoot.resolve("cgbn/backends/opencl/kernels")) {
        addsubKernelIncludes.forEach { include(it) }
        eachFile {
            path = name
        }
        includeEmptyDirs = false
    }
    into(layout.projectDirectory.dir("src/main/assets/kernels"))
}

val montsqrKernelIncludes = arrayOf(
    "mont_priv.cl",
    "mont_priv_opt.cl",
    "mont_mul_unroll_only_512_manual_generated.cl",
    "mont_priv_bench.cl",
    "mont_priv_opt_bench.cl",
    "mont_wg.cl",
    "mont_wg_bench.cl",
    "mont_mul_unroll_i24.cl",
    "mont_mul_unroll_i24_384_manual_generated.cl",
    "mont_mul_unroll_i24_bench.cl",
)

tasks.register<Copy>("syncMontsqrKernels") {
    from(mpaRoot.resolve("cgbn/backends/opencl/kernels")) {
        montsqrKernelIncludes.forEach { include(it) }
    }
    into(layout.projectDirectory.dir("src/main/assets/kernels/cgbn/backends/opencl/kernels"))
}

val ecmStage1KernelIncludes = arrayOf(
    "ecm_stage1.cl",
    "ecm_stage1_mont4096_paths.cl",
    "mp_addsub/stage1/asm_block32_stage1.cl",
    "mp_addsub/stage1/asm_block16_stage1.cl",
)

tasks.register<Copy>("syncEcmStage1Kernels") {
    from(mpaRoot.resolve("cgbn/backends/opencl/kernels")) {
        ecmStage1KernelIncludes.forEach { include(it) }
    }
    into(layout.projectDirectory.dir("src/main/assets/kernels/cgbn/backends/opencl/kernels"))
}

tasks.named("preBuild") {
    dependsOn("syncAddsubKernels", "syncAddsubKernelsFlat", "syncMontsqrKernels", "syncEcmStage1Kernels")
}

if (ecmAndroidGmpRoot != null) {
    logger.lifecycle("ECM: linking GMP from ecm.android.gmp.root=$ecmAndroidGmpRoot")
} else {
    logger.lifecycle(
        "ECM: ecm.android.gmp.root not set — factorization disabled; see docs/DEV_GMP_SETUP.md",
    )
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