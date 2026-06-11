# Android ECM：界面与设置开发说明

本文档说明 `Android/ECM` 应用壳层的布局结构、顶栏/底栏、设置项联动，以及常见 UI 修改入口。算子与 native 逻辑见同目录下其他 `DEV_*.md`。

---

## 1. 界面结构

```
activity_main.xml
├── AppBarLayout + MaterialToolbar     ← 标题、副标题、右上角 ⋮ 菜单
├── NestedScrollView
│   ├── panel_ecm  (content_ecm.xml)   ← ECM 页（默认）
│   └── panel_bench (content_bench.xml)← 基准页（visibility 切换）
├── BottomNavigationView               ← ECM / 基准
```

| 页面 | 布局 | 说明 |
|------|------|------|
| 主界面 | `activity_main.xml` | 双面板 + 底栏；日志卡片在各自 `content_*.xml` 底部 |
| ECM | `content_ecm.xml` | 性能、分解参数、诊断、输出日志 |
| 基准 | `content_bench.xml` | 微基准参数、算子按钮、输出日志 |
| 设置 | `activity_settings.xml` | 分组子标题 + `SwitchMaterial` 行 |
| 关于 | `activity_about.xml` | 版本与简介 |

`MainActivity.showTab()` 切换 `panel_ecm` / `panel_bench` 可见性，并更新 Toolbar 副标题（`subtitle_ecm` / `subtitle_bench`）。

---

## 2. 顶栏（Toolbar）

### 2.1 主标题与副标题

- 主标题：`app:title="@string/app_name"`（`ECM OpenCL`）
- 副标题：代码中 `toolbar.subtitle = getString(R.string.subtitle_ecm|subtitle_bench)`
- 副标题样式：`app:subtitleTextAppearance="@style/TextAppearance.ECM.Toolbar.Subtitle"`（`res/values/style.xml`）

`strings.xml` 中的 `subtitle`（「GPU OpenCL ECM 分解 & 微基准」）为历史遗留，**当前未绑定到界面**；实际显示的是 `subtitle_ecm` / `subtitle_bench`。

### 2.2 右上角溢出菜单（竖三点）

- 菜单资源：`res/menu/toolbar_menu.xml`（设置、关于）
- 布局：`activity_main.xml` 中 `app:menu="@menu/toolbar_menu"`
- 图标颜色：Toolbar 上 `android:theme="@style/ThemeOverlay.MaterialComponents.Dark.ActionBar"` → 溢出按钮为白色
- 弹出菜单主题：`app:popupTheme="@style/ThemeOverlay.ECM.ToolbarPopup"`
- 点击处理：`MainActivity.setupToolbarMenu()` → 打开 `SettingsActivity` / `AboutActivity`

### 2.3 刘海 / 状态栏安全区

`MainActivity.setupWindowInsets()`：`AppBarLayout` 增加 `statusBars + displayCutout` 的 top padding；底栏增加 `navigationBars` bottom padding。主题中 `statusBarColor` 为透明，紫色顶栏延伸到状态栏区域。

---

## 3. 设置页（SettingsActivity）

### 3.1 数据存储

> 重启/升级后哪些数据仍保留、OpenCL 缓存与日志路径等完整说明见 **[DEV_ANDROID_STORAGE.md](DEV_ANDROID_STORAGE.md)**。

`AppSettings.kt` 读写外部 `config/ecm_app_settings.xml`（见 [DEV_ANDROID_STORAGE.md](DEV_ANDROID_STORAGE.md)）：

| 键 | 类型 | 默认 | 说明 |
|----|------|------|------|
| `follow_system` | Boolean | `true` | 外观跟随系统 |
| `dark_mode` | Boolean | `false` | 手动深色（仅在不跟随时生效） |
| `log_to_file` | Boolean | `true` | 追加写入 `ecm.log` / `bench.log` |
| `log_saved_toast` | Boolean | `true` | 运行结束 Toast |

应用启动时 `EcmApplication` 调用 `AppSettings.applyTheme()`，内部使用 `AppCompatDelegate.setDefaultNightMode()`。

### 3.2 「跟随系统」开启时禁用「深色模式」

实现位置：`SettingsActivity.updateDarkModeEnabled()`。

```kotlin
private fun updateDarkModeEnabled() {
    val follow = switchFollowSystem.isChecked
    switchDarkMode.isEnabled = !follow          // 开关不可点
    findViewById<View>(R.id.row_dark_mode).alpha = if (follow) 0.5f else 1f  // 整行半透明
}
```

调用时机：

1. `bindFromPrefs()` 读完设置文件后；
2. 用户切换「跟随系统」开关后（`setupListeners` 内）。

逻辑含义：跟随系统时由系统决定深浅色，手动「深色模式」无意义，故 **仅禁用 UI**，不修改 `dark_mode` 已保存的值；用户再次关闭「跟随系统」后，之前的深色开关状态仍保留。

主题生效：`AppSettings.setFollowSystem` / `setDarkMode` → `applyTheme()` → `SettingsActivity.onThemePreferenceChanged()` → `recreate()`；返回主界面时若 `RESULT_OK` 也会 `recreate()`。

### 3.3 日志相关开关

#### 日志追加到文件（`log_to_file`）

- `MainActivity.beginLoggedSession()` / `appendToLog()`：若关闭则 **不写文件**
- `refreshLogUi()`：关闭时日志路径行显示 `log_path_disabled`

#### 日志写入 Toast（`log_saved_toast`）

- `MainActivity.notifyLogSaved()`：若关闭则不弹 Toast
- 与 `log_to_file` **独立存储**；关闭文件日志时 **不会** 自动把 Toast 开关置为 off

#### 「文件日志关闭时禁用 Toast 行」

`updateLogToastEnabled()` 仅在 UI 上禁用 Toast 开关（`isEnabled = false`、行透明度 0.5），**不改动** `log_saved_toast` 的偏好值。重新打开「日志追加到文件」后，Toast 开关恢复可点，并保持用户上次选择。

### 3.4 开关组件

设置页使用 `SwitchMaterial`（`com.google.android.material.switchmaterial.SwitchMaterial`），与 `Theme.MaterialComponents` 兼容。勿使用 `MaterialSwitch`（Material 3 专用，在当前主题下会 inflate 崩溃）。

`suppressCallbacks` 标志：在 `bindFromPrefs()` 批量赋值时避免触发 `OnCheckedChangeListener` 里的持久化/重建逻辑。

---

## 4. 日志文件

`RunLogStore.kt` 写入应用专属目录（无需存储权限）。路径、持久化与清理见 [DEV_ANDROID_STORAGE.md §2.2](DEV_ANDROID_STORAGE.md#22-日志文件runlogstore)。

每次运行以 `=== yyyy-MM-dd HH:mm:ss ===` 分段追加。路径显示在各自输出卡片底部 `log_path_ecm` / `log_path_bench`。

---

## 5. 深色模式资源

| 资源 | 浅色 | 深色 (`values-night/`) |
|------|------|-------------------------|
| `surface` | `#F8F9FC` | `#121318` |
| `card_bg` | `#FFFFFF` | `#1E1F25` |
| `text_primary` / `text_secondary` | 见 `values/colors.xml` | 见 `values-night/colors.xml` |

主题：`Theme.ECM` 父类为 `Theme.MaterialComponents.DayNight.NoActionBar`。

---

## 6. 常见修改指南

### 改顶栏副标题文案或字号

1. 文案：`res/values/strings.xml` → `subtitle_ecm` / `subtitle_bench`
2. 字号/颜色：`res/values/style.xml` → `TextAppearance.ECM.Toolbar.Subtitle`，或 Toolbar 上 `app:subtitleTextColor`

### 增加设置项

1. `activity_settings.xml`：在对应分组 `MaterialCardView` 内仿照现有行增加 `row_*` + `SwitchMaterial`
2. `strings.xml`：标题与 summary
3. `AppSettings.kt`：键、getter/setter
4. `SettingsActivity.kt`：`bindFromPrefs`、`setupListeners`；若有联动则增加 `update*Enabled()` 方法
5. 业务侧（如 `MainActivity`）读取 `AppSettings`

### 增加溢出菜单项

1. `res/menu/toolbar_menu.xml` 增加 `<item>`
2. `MainActivity.setupToolbarMenu()` 增加分支

### 新增独立子界面

1. `activity_*.xml` + `*Activity.kt`
2. `AndroidManifest.xml` 注册，`android:parentActivityName=".MainActivity"`
3. 复用 `setupWindowInsets()` 模式（见 `SettingsActivity` / `AboutActivity`）

---

## 7. 相关源文件索引

| 文件 | 职责 |
|------|------|
| `MainActivity.kt` | 主界面、底栏、菜单、运行与日志 UI |
| `SettingsActivity.kt` | 设置 UI 与开关联动 |
| `AppSettings.kt` | 偏好读写与主题 |
| `EcmApplication.kt` | 启动时应用主题 |
| `RunLogStore.kt` | 日志文件追加 |
| `activity_main.xml` | 主布局 |
| `content_ecm.xml` / `content_bench.xml` | 分页内容 |
| `activity_settings.xml` | 设置布局 |
| `menu/toolbar_menu.xml` | 顶栏溢出菜单 |
| `menu/bottom_nav.xml` | 底栏 |

存储与持久化总览见 [DEV_ANDROID_STORAGE.md](DEV_ANDROID_STORAGE.md)。
