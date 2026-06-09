# Android OpenCL minimal probe (ECM)

## 16 KB page size (Android 15+)

**Do not bundle `libOpenCL.so` in the APK.** Vendor libraries pulled via `adb` are usually
4 KB-aligned and will crash on 16 KB-page devices when packaged in `jniLibs/`.

The app loads OpenCL via **`uses-native-library` + `System.loadLibrary("OpenCL")`**.
Do not use absolute `/vendor/...` paths (linker namespace blocks them) and do not bundle
the `.so` in the APK (16 KB alignment).

`get_libOpenCL.bat` is for **inspection only** (objdump NEEDED / LOAD alignment).

## Build & run

1. Open `Android/ECM` in Android Studio
2. Ensure `jniLibs/` has **no** `libOpenCL.so` (delete if present)
3. Build **arm64-v8a**, run on real device
4. UI or `adb logcat -s ECM-OpenCL`

Success: `RESULT: PASS (OpenCL usable)`

## If OpenCL fails to load

- Device must expose `/vendor/lib64/libOpenCL.so` (most Adreno/Mali phones)
- Some OEMs block third-party OpenCL — check probe output for `dlopen fail`
