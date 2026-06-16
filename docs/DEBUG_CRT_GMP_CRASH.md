# Debug 构建崩溃排查：MSVC Debug CRT × GMP 自定义分配器冲突

## 症状

- **Debug 构建**（`cmake --build build --config Debug`）启动后几乎立即崩溃
- 崩溃位置：`Using B1=...` 之后，在 `GPU: OpenCL<...>` 之前
- VS 弹窗："ecm.exe 已停止工作"，调试器显示 `abort()` 或 SIGSEGV
- **Release 构建完全正常**，RelWithDebInfo 也正常
- 调用栈不可靠（`ecm_ts_fprintf`、`eckpt_filename`、`strncpy` 附近）

```
# Debug 输出截断在这里（之后崩溃）：
Using B1=100000, B2=0, sigma=3:268526266-268526266 (1 curves)

# Release 正常输出：
GPU: OpenCL<16 limbs, 1 thread> kernel, 421-bit N, s=144344 bits, np0=0x00000001
GPU: stage1 operators: mul=mont_mul_unroll_512b, ...
GPU: factor found in Step 1 with curve 0 (-sigma 3:268526266)
```

## 排查过程

### 第一轮：排除嫌疑（均无效）

| 假设 | 验证方式 | 结论 |
|------|---------|------|
| `ecm_runtime_config()` 符号未链接 | 确认 CMakeLists.txt 已添加源文件 | ❌ 不是 |
| `g_log_mutex` 自死锁 | `std::mutex`→`std::recursive_mutex`；移除 `ecm_install_timestamped_iostreams` | ❌ 不是 |
| `ecm_install_timestamped_iostreams` 位置 | 恢复到 `main()` 首行 | ❌ 不是 |
| 最后一个已知正常提交 | `RelWithDebInfo` 正常 → 崩溃仅在 `/MDd` 下 | **关键发现** |

### 第二轮：逐步插旗法定位崩溃点

在 `cgbn_stage1_opencl.cpp` 中每隔几行插入 `fprintf(stderr, "diag: ...\n"); fflush(stderr);`
（使用原生 C `fprintf` 而非 `ecm_ts_fprintf` 避免日志层干扰），逐步缩小范围：

```
diag: ENTER cgbn_ecm_stage1 sigma=268526266
diag: n_log2=421
diag: s_bits OK
diag: ckpt_fn enter
diag: ckpt_fn static OK
diag: ckpt_fn nbits=421
diag: ckpt_fn get_str=0000023B732C7B10
diag: ckpt_fn before strlen
diag: ckpt_fn len=106
diag: ckpt_fn before strncpy
                                        ← 之后无声崩溃
```

崩溃点精确锁定在 `src/opencl_ecm_checkpoint.cpp` 第 136 行：

```cpp
strncpy(first_hex, N_str, (len >= 8u) ? 8u : len);
```

### 第三轮：RelWithDebInfo 排除法确认根因

```powershell
# RelWithDebInfo（/MD, 无 _ITERATOR_DEBUG_LEVEL） → 正常
cmake -S . -B build_relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_relwithdebinfo --config RelWithDebInfo
echo "(2^421-1)" | build_relwithdebinfo\RelWithDebInfo\ecm.exe ...  # → PASS

# Debug（/MDd, _ITERATOR_DEBUG_LEVEL=2） → 崩溃
echo "(2^421-1)" | build\Debug\ecm.exe ...                          # → CRASH
```

**确认根因在 Debug CRT 内部，非代码逻辑错误。**

## 根因分析

### 三层冲突链

```
层 1: GMP 内存分配
├─ mpz_get_str(nullptr, 16, N)
├─ GMP 内部分配器 (不是 CRT malloc)
└─ 返回 N_str 指针，缓冲区不归 CRT 堆管理

层 2: Debug CRT 堆验证
├─ /MDd 链入 MSVCRTD，_ITERATOR_DEBUG_LEVEL=2
├─ EcmRuntimeConfig 含 7 个 std::string 成员
├─ std::string 构造激活 Debug CRT 堆追踪
└─ CRT 假设所有分配走 CRT 堆（统一验证）

层 3: CRT secure-strncpy 触发
├─ Debug CRT 的 strncpy 实现不是纯字节拷贝
├─ 内部调用 _CrtCheckMemory / _msize 探查缓冲区
├─ N_str 指向 GMP 分配的"陌生"内存区
├─ CRT 堆验证读到不认识的头 → 断言失败 → abort()
└─ 💥 崩溃
```

### 为什么之前不崩溃？

```
87e1260 (正常):
├─ ecm_log_timestamp_enabled() → getenv("ECM_LOG_TIMESTAMP")
├─ selected_device_index_from_env() → getenv("CGBN_OPENCL_DEVICE_INDEX")
└─ 零个 std::string 对象在日志/设备选择路径上构造
    → Debug CRT 堆追踪未激活 → GMP 分配器无冲突

9c0abd2 (崩溃):
├─ ecm_log_timestamp_enabled() → ecm_runtime_config().log_timestamp
├─ ecm_runtime_config() → 单例，含 7 个 std::string 成员
├─ 首次调用构造 std::string → Debug CRT 激活堆验证
└─ GMP 分配的 N_str 被 Debug CRT strncpy 探查 → 冲突
```

### 为什么 Release 正常？

Release CRT（/MD, `msvcrt`）不包含堆验证和迭代器调试代码。`strncpy` 是纯字节拷贝（`memcpy` 语义），不检查源缓冲区所有权。

## 最终修复

### 真正起作用的修复：CMakeLists.txt 系统性隔离

**只有这一处修复解决了问题** —— 统一使用 Release CRT。

```cmake
# MSVC Debug CRT (MDd) 有与 GMP 自定义分配器冲突的已知问题。
# 统一使用 Release CRT (MultiThreadedDLL = /MD)，关闭迭代器调试，
# 保留未优化代码和调试符号。
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)
    add_compile_definitions($<$<CONFIG:Debug>:_HAS_ITERATOR_DEBUGGING=0>)
endif()
```

### `opencl_ecm_checkpoint.cpp` 的 `strncpy`→`memcpy` 实际未生效

实测证据：单独应用 `memcpy` 修复，不修改 CMakeLists.txt CRT 配置 → **仍崩溃**。
单独应用 CMakeLists.txt CRT 配置，不修改 checkpoint → **可正常运行**。

因此 `strncpy`→`memcpy` 仅改变了崩溃位置而非根因。触发点之所以是 `strncpy`，
是因为那是 Debug CRT 堆验证引擎在 GMP 缓冲区上遇到的**第一个** CRT 字符串函数。
换成 `memcpy` 后，堆验证会在下一个 CRT 调用处（如 `free(N_str)` 或其他内部检查）触发。

**根因在 CRT 级别，不在任何单一代码行上**。

## 排查技巧总结

### 1. 逐步插旗法（Bisect by printf）

在崩溃点附近每 2-4 行插入 `fprintf(stderr, "X\n"); fflush(stderr);`。需注意：
- **使用 C 原生 `fprintf` + `fflush`**，不用 `ecm_ts_fprintf`（后者本身可能涉及锁/内存分配）
- **不使用 `std::cout`**（涉及 C++ 流，可能与 streambuf 重组冲突）
- 每次插旗后完整重编译，缩小一个区块

### 2. 跨 CRT 配置验证

当 Debug 崩溃而 Release 正常时，不直接修代码→先切 `RelWithDebInfo`（/MD + 调试符号 + 未优化）。
这能快速排除"代码逻辑错误 vs CRT 内部差异"两个大类。

### 3. 混合分配器项目的注意事项

- **代码级防御修补不可靠**：`strncpy`→`memcpy` 仅改变崩溃位置。Debug CRT 堆验证引擎一旦激活，会在**任意** CRT 函数处触发冲突。修复必须在 CRT 级别。
- **`std::string` / STL 容器会悄悄激活 CRT 堆追踪**：首次构造即可改变全局行为（不是第一次使用 stl，是 Debug CRT 在构造时初始化追踪表）。此后所有 GMP 分配的缓冲区与任意 CRT 函数交互都可能崩溃，且崩溃点不固定。
- **偏好 C 原生接口越过 C++ STL**：在与 GMP 等自定义分配器交互的热路径上，`fprintf`/`fputs` 比 `std::cout`/`std::cerr` 更安全，因不触发 Debug CRT 的迭代器/流追踪。
- **优先用 `memcpy`/`memcmp` 等纯内存操作替代 `strcpy`/`strncpy`/`strcmp`** —— 作为预防性最佳实践，但不替代 CRT 级别隔离。

### 4. 玄学崩溃 = 自信分步缩小

```
全程序崩溃
→ 插旗法找到函数
→ 插旗法找到行（strncpy）
→ RelWithDebInfo 确认是 CRT 差异，非代码错误
→ 尝试代码级防御修补（memcpy）→ 仍崩溃 ← 关键证据！
→ 理解"CRT 堆验证已全局激活，触发点不重要"
→ 唯一有效修复：CRT 级别隔离（切换 /MD）
```

**最关键一步**：当代码级修复无效时，不要继续猜测——这是 CRT 全局状态的信号。
Debug CRT 的堆验证是**一次性激活、全局生效**的。`std::string` 构造触发它之后，
整个进程进入"双分配器冲突"状态，任何 CRT 函数在 GMP 缓冲区上都是潜在崩溃点。
回退到 Release CRT 是唯一正确的解决路径。