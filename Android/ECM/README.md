# ECM OpenCL Android 开发文档

在真机上探测 GPU OpenCL，并运行与桌面版同源的 **ECM add/sub** 微基准（`opencl_ecm_addsub` 内核族）。

## 环境要求

| 项目 | 要求 |
|------|------|
| Android Studio | 近期稳定版，NDK r26+ |
| 设备 | **arm64-v8a** 真机，厂商提供 `libOpenCL.so`（常见 Adreno / Mali） |
| minSdk | 31（`uses-native-library` 加载系统 OpenCL） |
| 仓库位置 | 本模块位于 `MPA-OpenCl/Android/ECM`，构建时需能访问上级目录中的 OpenCL 内核与 manifest |

## 基本用法

1. 用 Android Studio 打开 **`Android/ECM`**（不是仓库根目录）。
2. 确认 **`jniLibs/` 内没有** 从手机 `adb pull` 的 `libOpenCL.so`（16 KB 页设备会因对齐错误崩溃）。
3. 连接真机，选择 **arm64-v8a** 构建并 Run。
4. 应用启动后会自动做一次 **设备探测**；结果在底部等宽文本区。

### 界面功能

| 区域 | 说明 |
|------|------|
| **设备性能**（顶部卡片） | 独立后台线程约每 1.5s 刷新：GPU 占用/频率优先读 **高通系统属性**（`vendor.gpu.*`，不受 SELinux 限制），再回退 sysfs；显示数据源 |
| 设备探测 | 枚举平台/设备、读写 buffer 冒烟测试 |
| 简短测试 | GPU 名称 + buffer ping |
| **ECM add/sub 微基准** | 与桌面 `opencl_ecm_addsub.exe` 同参的 4 项可编辑参数 |
| Limb add-mod 微基准 | 独立的小型 16/24/32-bit 单内核测试（非 ECM 完整路径） |

### add/sub 四个参数

与 Windows 命令行一一对应：

```text
opencl_ecm_addsub.exe [--bits <bits>] <kernel_iterations> <instances> <launch_repeats>
```

| UI 字段 | 含义 | 默认（手机友好） | 桌面常见示例 |
|---------|------|----------------|--------------|
| bits | 模数位宽，须为 32 的倍数 | 512 | 512 / 4096 |
| kernel_iterations | 单次 launch 内核内循环次数 | 1000 | 10000 |
| instances | 并行实例数（global size） | 64 | 128 |
| launch_repeats | 重复 enqueue 次数 | 3 | 3 |

总运算量：`instances × kernel_iterations × launch_repeats`。

输出中 **ms** 为固定小数；**ops/s ≥ 1e6** 时使用科学计数法（与 Windows `opencl_ecm_addsub` 一致，例如 `1.2224e+07 ops/s`）。

## 与 Windows 的复用关系

```text
MPA-OpenCl/
├── cgbn/backends/opencl/kernels/          ← 复用：OpenCL 内核源码
│   ├── ecm_addsub_bench.cl
│   └── mp_addsub/generated/*.cl
├── include/opencl_ecm_addsub_manifest.h   ← 复用：内核列表与构建清单 API
├── src/opencl_ecm_addsub_manifest.cpp     ← 复用：manifest 实现（CMake 链入 libecm.so）
└── src/opencl_ecm_addsub_bench.cpp        ← 桌面专用（GMP + cgbn::opencl），Android 不链接
```

### 从 Windows 复用的部分

| 资源 | 用途 |
|------|------|
| `ecm_addsub_bench.cl` | 基础 add/sub/mod 内核与 bench entry |
| `mp_addsub/generated/*.cl` | `fused_unroll`、`fused_unroll_auto`、priv 等展开实现 |
| `opencl_ecm_addsub_manifest.*` | 决定编译拼接哪些 `.cl`、bench 跑哪些 kernel 名 |
| 内核算法与路径命名 | `fused_unroll`、`fused_unroll_auto`、`legacy`、`mask` 等与桌面一致 |

Gradle 任务 **`syncAddsubKernels`** 在每次 `preBuild` 时把上述内核复制到：

`app/src/main/assets/kernels/cgbn/backends/opencl/kernels/`

也可手动执行：

```bash
./gradlew :app:syncAddsubKernels
```

### Android 独立实现的部分

| 组件 | 说明 |
|------|------|
| `app/src/main/cpp/opencl_loader.*` | `dlopen` 系统 `libOpenCL.so`，不依赖 Khronos 安装版 ICD |
| `app/src/main/cpp/opencl_probe.cpp` | 探测与简短测试 |
| `app/src/main/cpp/opencl_bench.cpp` | 16/24/32-bit limb 微基准（教学用，非 ECM manifest） |
| `app/src/main/cpp/opencl_addsub_bench.cpp` | ECM add/sub bench 宿主逻辑：读 assets、编译、计时、输出 |
| `app/src/main/cpp/kernel_assets.cpp` | 从 APK `AssetManager` 加载 `.cl` 文本 |
| `MainActivity.kt` + `activity_main.xml` | UI、参数输入、后台线程调 JNI |
| `AndroidManifest.xml` | `uses-native-library libOpenCL.so`；不打包 vendor `.so` |

与桌面的主要差异：

- **无 GMP**：正确性用主机端多精度 limb 例程校验 `ecm_mp_add_mod_fused`。
- **无 AMD GPU asm 路径**：`asm_enabled=false`，不编译 `mp_addsub/asm_*.cl`（手机为 Adreno/Mali）。
- **OpenCL 运行时**：动态加载 + 自声明 `cl_*` 常量；桌面用 `cgbn::opencl` + 链接 `OpenCL.lib`。
- **无** `opencl_ecm_montsqr` / mont mul/sqr bench（仍在桌面 `opencl_ecm_montsqr.exe`）。

## 工程结构（简要）

```text
Android/ECM/
├── app/
│   ├── build.gradle.kts      # syncAddsubKernels、abiFilters、禁止打包 libOpenCL
│   ├── src/main/
│   │   ├── AndroidManifest.xml
│   │   ├── assets/kernels/...   # 构建时同步的内核（勿手改，会被覆盖）
│   │   ├── cpp/
│   │   │   ├── CMakeLists.txt   # 链接 manifest.cpp，MPA_ROOT 指仓库根
│   │   │   ├── native-lib.cpp   # JNI
│   │   │   └── opencl_addsub_bench.cpp
│   │   ├── java/.../MainActivity.kt
│   │   └── res/layout/activity_main.xml
└── README.md                 # 本文档
```

## 构建注意

- **不要** 将 `adb pull` 的 `libOpenCL.so` 放入 `jniLibs`。
- 修改仓库内 `cgbn/.../kernels` 后需重新构建 APK（或先跑 `syncAddsubKernels`）。
- 大位宽（如 4096-bit）首次 OpenCL 编译可能需数十秒至数分钟，属正常现象。
- 日志：`adb logcat -s ECM-OpenCL`

## GPU 占用 / 频率（Adreno）

读取顺序：

1. **sysfs**（`kgsl/gpubusy`、`gpuclk`）优先——Adreno 642 高负载时准确；空闲时可能读不到（显示 `—`）。
2. **系统属性白名单**（仅 `vendor.gpu.freq` 等显式列表，**不**扫描全表，避免 MIUI `persist.sys.computility.gpulevel` 误当占用率）。
3. 当前频率不可读时，尝试显示 **max_gpuclk** 为 `— (max xxx MHz)`。

8 Elite (830) 上 `getprop` 无 `vendor.gpu.freq`，且 sysfs 常被 SELinux 拒绝，占用/频率可能长期为 `—`，属机型限制。

面板会显示 `GPU 数据源: 占用=系统属性, 频率=系统属性` 等来源标签。

若仍为 `—`，在电脑上查找本机属性名并反馈或自行加入 `sys_gpu_stats.cpp` 的 `kFreqProps`：

```bash
adb shell getprop | grep -i gpu
adb logcat -s ECM-GPUStats
```

## 输出示例

```text
=== ECM add/sub microbench ===
512-bit, kernel_iterations=10000, instances=128, launch_repeats=1
device: QUALCOMM Adreno(TM) 830
compile: 227.663 ms
verify ecm_mp_add_mod_fused: PASS

--- mp_add_mod ---
fused_unroll: 104.709 ms, 1.2224e+07 ops/s
...

RESULT: PASS
```

## 相关桌面工具

| 可执行文件 | 作用 |
|------------|------|
| `build/Debug/opencl_ecm_addsub.exe` | 完整 ECM add/sub bench（GMP 校验、ASM、LPT 等） |
| `build/Debug/opencl_ecm_montsqr.exe` | Montgomery mul/sqr bench（Android 未移植） |
| `build/Debug/ecm.exe` | 完整 ECM stage1 驱动 |

更上层的 Android 总览见 [`../README.md`](../README.md)。
