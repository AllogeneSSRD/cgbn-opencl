# ECM 日志解析器 (ECM Log Parser)

一个可视化的 Prime95 / mprime worker 日志解析器。上传日志文件，自动将每一次 ECM 运行解析为一行数据，支持筛选、时间线可视化，并可导出 XLSX / CSV。

## 功能

- **网页上传** `.log` / `.txt` 文件，后端 (Flask + Python) 解析。
- **运行分段**：每一条 `ECM on M...: Edwards curve #N` 开启一次运行，直到该 worker 的下一条 `ECM on`（或文件结尾）为止。重启/中断都会各自成为一行，并用状态列标记：
  - `complete` — 完整跑完 Stage 1 & Stage 2
  - `stage1-only` — 只完成 Stage 1
  - `interrupted` — Stage 1 未完成
- **提取字段**：Worker、指数、Curve#、s、B1、Actual B2、Worth、Available/Using 内存、Stage 1 时间、Stage 2 init/complete/GCD 时间、S1 FFT、S2 FFT、FFT 类型、开始/结束时间。
  - S1 FFT = ECM 行之前最近的 `Using ... FFT length`。
  - S2 FFT = Stage 1 完成后到 Stage 2 之间的 `Switching to ... FFT length`（若无则沿用 S1）。
- **筛选**：Worker 序号 / 指数 / Curve# / 状态（多选），B1 / Available mem / Using mem（区间），日期范围（按开始时间）。
- **可视化**：汇总统计卡片、按 Worker 分组的 Gantt 时间线（Stage 1 蓝 / Stage 2 橙）、可排序的数据表格。
- **导出**：XLSX（openpyxl）或 CSV，可选“当前筛选”或“全部”。

## 运行

```bash
conda activate web
cd D:\code\MPA-OpenCl\tools\log_parser
pip install -r requirements.txt   # 首次运行
python app.py
```

然后浏览器打开 http://127.0.0.1:8000 （如需换端口：`set PORT=8080` 后再运行）。

## 命令行快速验证解析器

```bash
python parser.py ..\screen_example.log
```

## 文件结构

```
log_parser/
├── app.py              # Flask 服务与导出
├── parser.py           # 日志解析核心 + 列定义
├── requirements.txt
├── templates/index.html
└── static/
    ├── app.js          # 前端逻辑（筛选/表格/Gantt/导出）
    └── style.css
```
