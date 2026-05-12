# 📑 重构文档索引 —— 第一步完成

## 🗺️ 快速导航

| 我想要... | 查看文件 | 阅读时间 |
|----------|--------|--------|
| **快速了解现在的状况** | [`QUICK_START.md`](QUICK_START.md) | 5 分钟 |
| **理解 CGBN 的整体架构** | [`ARCHITECTURE.md`](ARCHITECTURE.md) | 15 分钟 |
| **找某个算子的调用链** | [`OPERATOR_DISPATCH.md`](OPERATOR_DISPATCH.md) | 5 分钟（查表） |
| **理解某个 CUDA 特性如何替换** | [`OPERATOR_DISPATCH.md` 附录](OPERATOR_DISPATCH.md#附录cuda-依赖点详解) | 5-10 分钟 |
| **了解接口冻结的细节** | [`STEP_1_INTERFACE_FREEZE.md`](STEP_1_INTERFACE_FREEZE.md) | 30 分钟 |
| **查看第一步的完成总结** | [`COMPLETION_SUMMARY.md`](COMPLETION_SUMMARY.md) | 10 分钟 |
| **了解下一步计划** | [`STEP_2_PLAN.md`](STEP_2_PLAN.md)（待创建） | - |

---

## 📂 文件清单

### 核心规范文档

#### 1. **QUICK_START.md** (必读)
- **长度**: ~400 行
- **用途**: 快速启动和全景导航
- **关键内容**:
  - 目录结构总览
  - 核心文档说明
  - 第一步成果
  - 下一步计划
  - 常见问题解答
- **适合人群**: 所有人（第一次接触）

#### 2. **ARCHITECTURE.md** (必读)
- **长度**: ~350 行
- **用途**: 理解设计和分层架构
- **关键内容**:
  - 整体布局（后端隔离结构）
  - 四层分层职责详解
  - 第一步任务
  - 输出物清单
  - 快速开始指南
- **适合人群**: 需要理解设计的人

#### 3. **OPERATOR_DISPATCH.md** (重要)
- **长度**: ~500 行
- **用途**: 算子分类、链路追踪、迁移参考
- **关键内容**:
  - 低风险算子（19 个）详表
  - 中风险算子（20 个）详表
  - 高风险算子（30+ 个）详表
  - 累加器和内存操作
  - 快速参考（按阶段分类）
  - **附录 A**: CUDA 依赖点详解（shuffle、ballot、carry chain 等）
- **适合人群**: 需要实现算子的开发者

#### 4. **STEP_1_INTERFACE_FREEZE.md** (参考)
- **长度**: ~450 行
- **用途**: 接口冻结的完整清单和验证方法
- **关键内容**:
  - 部分 A: 公共 API 层审查（错误报告、模板函数、数据类型）
  - 部分 B: 后端适配层审查（context_t、env_t、内存类型）
  - 部分 C: 后端依赖点清单（8 大类依赖）
  - 部分 D: 算子调度映射（所有 ~60+ 个方法）
  - 部分 E-H: 行动项、验证方法、进度跟踪
- **适合人群**: 需要深入理解接口的人

#### 5. **COMPLETION_SUMMARY.md** (总结)
- **长度**: ~300 行
- **用途**: 第一步的成果总结和快速参考
- **关键内容**:
  - 已完成工作列表
  - 分析成果总结（API、接口、分类、依赖、路线图）
  - 关键发现
  - 文档使用建议
  - 快速参考（常见问题）
  - 第二步预览
  - 验证清单
- **适合人群**: 需要了解全局进度的人

---

## 🎯 典型使用场景

### 场景 1：入职新员工，需要快速了解项目
```
1. 读 QUICK_START.md (5 分钟)
   ↓ 了解全景、文档结构、核心概念
2. 读 ARCHITECTURE.md 的"分层职责" (10 分钟)
   ↓ 理解为什么这样设计、后端隔离的意义
3. 扫一眼 OPERATOR_DISPATCH.md 的三个主表格 (5 分钟)
   ↓ 了解算子分类和风险等级
→ 总用时: 20 分钟，获得足够上手的知识
```

### 场景 2：需要实现某个算子（如 cgbn_add）的 OpenCL 版本
```
1. 打开 OPERATOR_DISPATCH.md，第三部分（高风险）
   → 查到 cgbn_add，依赖: carry chain + threadIdx resolve
2. 打开 OPERATOR_DISPATCH.md 附录 A4（Carry Chain 依赖）
   → 查看 CUDA 代码示例和 OpenCL 替换策略
3. 查看 STEP_1_INTERFACE_FREEZE.md 部分 C2
   → 确认 carry chain 出现在哪些文件
4. 打开 cgbn/core/core_add_sub.cu，查看 carry 链式调用
5. 参照示例，在 OpenCL kernel 中实现 portable carry
→ 总用时: 30-60 分钟，获得完整的迁移方案
```

### 场景 3：需要理解 CUDA 特性在 OpenCL 中怎么处理
```
# 问题：CUDA 的 __shfl_sync() 在 OpenCL 中怎么实现？
1. 打开 OPERATOR_DISPATCH.md 附录 A2（Shuffle 依赖）
   → 查看 CUDA 模式、出现位置、OpenCL 替换、代码示例
2. 如果需要更多上下文，打开 STEP_1_INTERFACE_FREEZE.md 部分 C
   → 确认 shuffle 的风险等级、影响范围
3. 决定是用 subgroup extension 还是 local memory 方案
→ 总用时: 15 分钟，获得清晰的替换方案
```

### 场景 4：追踪某个高风险的复合操作（如 Montgomery 乘法）
```
1. 打开 OPERATOR_DISPATCH.md，第三部分
   → 查到 cgbn_mont_mul，优先级 P5，依赖: __shfl_sync, carry chain, mont algo
2. 打开 OPERATOR_DISPATCH.md 附录 A2 和 A4
   → 逐个查看 shuffle 和 carry chain 的 OpenCL 方案
3. 打开 cgbn/core/core_mont.cu，查看 Montgomery 算法
4. 打开 mpaKernel_32bits.cl，查看现有 OpenCL Montgomery 实现（参考）
5. 比对 CUDA 版和现有 OpenCL 版，设计新的 OpenCL backend 实现
→ 总用时: 1-2 小时，获得完整的迁移规划
```

---

## 📊 文档内容对应关系

```
ARCHITECTURE.md
├── 整体布局 ─────────→ 理解目录结构
├── 分层职责 ─────────→ 理解四层设计
├── 第一步任务 ──────→ 了解本阶段目标
└── 输出物清单 ──────→ 参考交付物

STEP_1_INTERFACE_FREEZE.md
├── 公共 API 层 ────→ 冻结的模板函数
├── 后端适配层 ────→ 必须实现的接口
├── 后端依赖点 ────→ CUDA 特性分类
├── 算子映射 ──────→ 调用链路
└── 行动项 ────────→ 代码审查检查表

OPERATOR_DISPATCH.md
├── 低风险算子 ────→ 可快速移植的 19 个
├── 中风险算子 ────→ 需部分适配的 20 个
├── 高风险算子 ────→ 需重设计的 30+ 个
├── 快速参考 ──────→ 按优先级分类
└── 附录 A ────────→ CUDA 依赖点详解

COMPLETION_SUMMARY.md
├── 已完成工作 ────→ 第一步成果
├── 分析成果 ──────→ 量化结果
├── 关键发现 ──────→ 项目瓶颈和机遇
└── 第二步预览 ────→ 下一阶段方向
```

---

## ✨ 快速查询表

### 我需要查找...

| 需求 | 查看位置 |
|------|--------|
| 某个算子的调用链 | `OPERATOR_DISPATCH.md` → 第一/二/三部分 → 按算子名查表 |
| 某个算子的风险等级 | `OPERATOR_DISPATCH.md` → 风险列 |
| 某个 CUDA 特性的替换方案 | `OPERATOR_DISPATCH.md` → 附录 A → 特性名 |
| 所有低风险算子 | `OPERATOR_DISPATCH.md` → 第一部分 → 19 个 |
| OpenCL 后端需实现的接口 | `STEP_1_INTERFACE_FREEZE.md` → B.2 |
| 公共 API 的完整列表 | `STEP_1_INTERFACE_FREEZE.md` → A.2 |
| CUDA 依赖分布统计 | `STEP_1_INTERFACE_FREEZE.md` → C.2 |
| 算子迁移优先级 | `OPERATOR_DISPATCH.md` → 第三部分 → 优先级列 |
| 推荐阅读顺序 | `QUICK_START.md` → 如何使用这些文档 |

---

## 🔗 跨文档参考链接

### 从 ARCHITECTURE 跳转
- "第一步任务" → `STEP_1_INTERFACE_FREEZE.md`
- "输出物清单" → 列出的每个文件

### 从 STEP_1_INTERFACE_FREEZE 跳转
- "CUDA 特性" → `OPERATOR_DISPATCH.md` 附录
- "算子映射" → `OPERATOR_DISPATCH.md` 主表格

### 从 OPERATOR_DISPATCH 跳转
- "低风险算子" → 第一部分表格
- "迁移策略" → 附录 A 对应小节

---

## 📚 推荐学习路径

### 路径 A：全面理解（2-3 小时）
1. QUICK_START.md (5 min)
2. ARCHITECTURE.md (20 min)
3. OPERATOR_DISPATCH.md (30 min 专注第一部分)
4. STEP_1_INTERFACE_FREEZE.md (60 min)
5. COMPLETION_SUMMARY.md (10 min)

### 路径 B：快速上手（30 分钟）
1. QUICK_START.md (5 min)
2. ARCHITECTURE.md 的"分层职责" (10 min)
3. OPERATOR_DISPATCH.md 的表格快速浏览 (10 min)
4. QUICK_START.md 的"常见问题" (5 min)

### 路径 C：针对性学习（15-30 分钟）
1. 确定你的角色（架构师/开发者/测试）
2. 从快速参考表找相关文档
3. 用 Ctrl+F 查找关键词
4. 按需深入相关章节

---

## ⚙️ 文档维护和更新

### 何时更新
- [ ] 实现第二步后，更新所有文档中的"步骤 2 预览"
- [ ] 发现冻结的接口有遗漏时，立即更新 `STEP_1_INTERFACE_FREEZE.md`
- [ ] 算子迁移遇到问题时，更新 `OPERATOR_DISPATCH.md` 的替换策略
- [ ] 完成每个阶段后，更新 `COMPLETION_SUMMARY.md` 的进度

### 更新优先级
1. 🔴 **高**: 接口规范、依赖点映射（影响下游）
2. 🟡 **中**: 快速参考、阅读建议
3. 🟢 **低**: 示例代码、详细说明

---

## 📞 常见问题

**Q: 文档太多了，我从哪里开始？**  
A: 从 `QUICK_START.md` 开始，它会告诉你其他文档的用途。

**Q: 我只需要实现一个算子，需要读所有文档吗？**  
A: 不需要。按场景 2（需要实现某个算子）的路径，只需 30-60 分钟。

**Q: 哪个文档更新最频繁？**  
A: 预期是 `OPERATOR_DISPATCH.md` 和 `STEP_1_INTERFACE_FREEZE.md`，因为涉及具体实现细节。

**Q: 文档和代码怎么同步？**  
A: 建议在 git commit 中引用相关文档，在 PR description 中说明哪些文档需要更新。

---

## 🎓 相关资源

- **原始计划**: `.github/prompts/plan-cgbnOpenCl.prompt.md`
- **CGBN 源代码**: `cgbn/` 目录
- **现有 OpenCL 参考**: `mpa_*.c`, `mpaKernel_*.cl`
- **OpenCL 标准**: https://www.khronos.org/opencl/
- **CUDA 文档**: https://docs.nvidia.com/cuda/

---

**🚀 准备好深入了吗？选择上面任何一个文档开始阅读吧！**

