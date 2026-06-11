# Android ECM：数据存储与持久化

本文说明 ECM 应用使用的各类存储位置、重启/升级后是否保留，以及如何查看与清理。UI 设置项含义见 [DEV_ANDROID_UI.md](DEV_ANDROID_UI.md)。

应用包名：`com.example.ecm`。

路径统一由 `AppStoragePaths.kt` 解析：在**应用专属外部目录** `Android/data/com.example.ecm/` 下与 `files/` 同级存放可调试数据（无需存储权限）。

---

## 1. 外部应用目录布局

```
/sdcard/Android/data/com.example.ecm/
├── config/
│   └── ecm_app_settings.xml    ← 设置（外观 / 日志开关）
├── logs/
│   ├── ecm.log
│   └── bench.log
└── opencl_cache/
    └── opencl_<hash>.bin       ← OpenCL 编译磁盘缓存
```

Android 11+ 部分文件管理器需对「Android/data」授权后才可浏览上述目录。

若外部存储不可用，上述内容回退到**内部** `context.filesDir/` 下同名子目录（文件管理器不可见）。

---

## 2. 总览

| 数据 | 机制 | 典型路径 | 重启 | 覆盖升级 | 卸载 |
|------|------|----------|------|----------|------|
| 设置 | XML 文件 | `.../config/ecm_app_settings.xml` | 保留 | **保留** | 删除 |
| 运行日志 | 追加写入 | `.../logs/*.log` | 保留 | **保留** | 删除 |
| OpenCL 磁盘缓存 | native 写入 | `.../opencl_cache/*.bin` | 保留 | **通常保留**¹ | 删除 |
| OpenCL RAM 缓存 | 进程内存 | — | 不保留 | — | — |
| 内核 `.cl` | APK assets | APK 内只读 | 随 APK | **随 APK 更新** | — |
| 主界面输入框 | 仅内存 | — | **不保留** | 不保留 | — |

¹ 内核或编译选项变更时旧 `.bin` 可能 cache miss，会重新编译；旧文件可手动删除。

---

## 3. 设置（`AppSettings`）

- **实现**：读写 `AppStoragePaths.settingsFile()`，格式与 Android `SharedPreferences` 导出 XML 相同：
  ```xml
  <?xml version='1.0' encoding='utf-8' standalone='yes' ?>
  <map>
      <boolean name="follow_system" value="true" />
      ...
  </map>
  ```
- **键**：`follow_system`、`dark_mode`、`log_to_file`、`log_saved_toast`
- **迁移**：首次启动若外部 XML 不存在、但内部仍有旧版 `shared_prefs/ecm_app_settings.xml`，会自动迁移到外部 `config/` 并清空旧 prefs。

---

## 4. 日志（`RunLogStore`）

- **目录**：`AppStoragePaths.logsDir()`
- 每次运行以 `=== yyyy-MM-dd HH:mm:ss ===` 分段追加。

---

## 5. OpenCL 编译缓存

- **初始化**：`MainActivity` → `nativeInitAssets(assets, AppStoragePaths.openClCacheRoot())`
- **native**：`set_opencl_cache_dir(root)`，实际缓存目录为 `{root}/opencl_cache/`（见 `opencl_program_cache.cpp`）
- **ECM 运行**：`ecm_android_run.cpp` 通过 `CGBN_OPENCL_CACHE_DIR` 指向同一目录
-  bench / probe 输出中的 `cache_dir`、`cache_key` 可直接对照文件管理器中的路径

---

## 6. 查看与清理

### 6.1 文件管理器 / adb（推荐调试）

```bash
# 设置
adb shell cat /sdcard/Android/data/com.example.ecm/config/ecm_app_settings.xml

# 日志
adb pull /sdcard/Android/data/com.example.ecm/logs/

# OpenCL 缓存
adb shell ls -la /sdcard/Android/data/com.example.ecm/opencl_cache/
```

### 6.2 系统设置

| 操作 | 影响 |
|------|------|
| **清除数据** / 卸载 | 删除内部数据 + `Android/data/com.example.ecm/` 全部（设置、日志、OpenCL 缓存） |
| **清除缓存** | 主要影响内部 `cache/`、`code_cache/`；**外部** `Android/data/.../config|logs|opencl_cache` 通常仍在 |

---

## 7. 实现索引

| 组件 | 文件 |
|------|------|
| 路径解析 | `AppStoragePaths.kt` |
| 设置读写 | `AppSettings.kt` |
| 日志 | `RunLogStore.kt` |
| 缓存根传入 native | `MainActivity.kt` → `nativeInitAssets` |
| OpenCL 缓存逻辑 | `opencl_program_cache.cpp` |

---

## 8. 与 UI 文档

见 [DEV_ANDROID_UI.md §3.1](DEV_ANDROID_UI.md#31-数据存储)。
