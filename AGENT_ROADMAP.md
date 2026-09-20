# D2INV → Agent 改造路线图

## Phase 1 — 最小可行 Agent ✅ 已完成

**目标**: 把硬编码流水线替换为 LLM 驱动的动态决策循环（ReAct 模式）

### 已实现
| 组件 | 文件 | 说明 |
|---|---|---|
| Agent 核心 | `agent/core.py` | 主循环：Think→Act→Observe→Decide，流式输出 |
| 规划器 | `agent/planner.py` | LLM 决策下一步该调用哪个工具 |
| 会话管理 | `agent/session.py` | 短期记忆：对话历史 + 流水线中间状态 |
| 工具接口抽象 | `agent/tools/base.py` | BaseTool / ToolRegistry / Action / ActionResult |
| 数据摘要工具 | `agent/tools/summarize.py` | 封装 file_summary + list_datasets |
| 故事生成工具 | `agent/tools/story.py` | 封装 DataStory.run_4r() |
| 模板生成工具 | `agent/tools/template.py` | 封装 InfographicTemplate.run() |
| 图表生成工具 | `agent/tools/charts.py` | 封装 visualize_data_story() |
| INV 组装工具 | `agent/tools/assemble.py` | 组合模板 + 图表 + 数据 |
| 自评工具 | `agent/tools/evaluate.py` | 5 维度评估雷达图 |

---

## Phase 2 — 工具使用扩展 ✅ 已完成

**目标**: 扩展核心工具集，添加通用 Agent 能力（Python REPL、网页搜索、数据库、文件系统、Shell）

### 已实现
| 组件 | 文件 | 说明 |
|---|---|---|
| Python REPL 工具 | `agent/tools/python_repl.py` | 沙箱化 Python 执行（subprocess 隔离、超时、受限 imports） |
| Web Search 工具 | `agent/tools/web_search.py` | DuckDuckGo HTML 搜索后端（免 API key） |
| SQL Database 工具 | `agent/tools/database.py` | 只读 SQLite 查询 + 表结构查看 |
| File System 工具 | `agent/tools/file_system.py` | 安全文件读写（工作区限制、路径穿越防护） |
| Shell Command 工具 | `agent/tools/shell.py` | 白名单 Shell 命令执行 |

---

## Phase 3 — 完整 Agent ✅ 刚刚完成

**目标**: 长期记忆、智能错误恢复、用户偏好学习

### 已实现
| 组件 | 文件 | 说明 |
|---|---|---|
| 文本嵌入器 | `agent/memory/embedder.py` | 字符 n-gram 哈希嵌入（无需外部模型） |
| 向量存储 | `agent/memory/vector_store.py` | numpy 余弦相似度搜索 + JSON 持久化 |
| 长期记忆管理器 | `agent/memory/long_term.py` | 跨会话语义记忆（对话/工具结果/事实存储与检索） |
| 错误分类器 | `agent/errors/classifier.py` | 6 类错误自动分类（瞬态/输入错误/缺失依赖/权限/逻辑/资源耗尽） |
| 恢复执行器 | `agent/errors/recovery.py` | 分层恢复策略（重试→纠正输入→降级→询问→中止） |
| 用户偏好学习 | `agent/preferences/learner.py` | 隐式偏好追踪（语言/图表类型/数据集） + 长期持久化 |

### 架构概览

```
用户输入
    │
    ▼
┌──────────────┐    ┌──────────────────┐
│  Agent Core  │◄───│ Long-Term Memory │  ← 语义回忆注入
│ (Think→Act→  │    │  (向量检索)       │
│  Observe→    │    └──────────────────┘
│  Remember)   │    
└──────┬───────┘    ┌──────────────────┐
       │       ◄────│ Error Recovery   │  ← 自动重试/降级
       │            └──────────────────┘
       │            ┌──────────────────┐
       │       ◄────│ User Preferences │  ← 偏好学习
       │            └──────────────────┘
       ▼
  ┌──────────────────────────────────┐
  │          Tool Registry           │
  │  (13 tools: Phase 1+2 combined) │
  └──────────────────────────────────┘
```

---

## Phase 4 — 进阶（待开始）

- 多 Agent 协作（Planning / Coding / Design / Review Agents）
- 可视化渲染效果自动验证
- 实时数据源监控
- 多轮迭代式优化
- Agent 间通信协议