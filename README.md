# 图片溯源智能体

基于 NiceGUI 的图片传播链溯源系统。上传一张图片，自动完成以图搜图 → 相似度校验 → 内容提取 → 传播分析 → 报告生成全流程。

## 项目结构

```text
project_root/
├── core/
│   ├── __init__.py
│   ├── state.py              # AgentState 定义、共享常量
│   └── visualization.py      # G6 拓扑图 HTML 生成
├── agents/
│   ├── __init__.py
│   ├── retriever.py          # 以图搜图（百度/Yandex/Google/Tineye/SauceNAO/SerpApi/Mitmproxy）
│   ├── validator.py          # 视觉校验（pHash + CLIP + OCR + 多模态 LLM 去重）
│   ├── analyzer.py           # 内容提取 & 传播分析（Firecrawl + LLM + Tikomni 微博/小红书 API）
│   └── orchestrator.py       # 编排入口 & 报告生成
├── tools/
│   ├── llmscrapy/            # LLM 网页抓取管线（多 fetcher 架构）
│   │   ├── fetcher.py        # Direct / Jina / Firecrawl 抓取器
│   │   ├── parser.py         # HTML → 结构化文本
│   │   ├── extractor.py      # LLM 字段提取
│   │   ├── pipeline.py       # 抓取→解析→提取 流水线
│   │   ├── stats_enricher.py # 百度互动量 API 补充
│   │   └── ...
│   └── mitmproxy/            # 代理抓包（微博/小红书搜索）
├── app.py                    # NiceGUI 主页面（浅色科技风）
├── workflow.py               # LangGraph 工作流（CLI 独立测试用）
├── requirements.txt
├── .env.example
└── README.md
```

## 模块职责

| 模块 | 职责 |
|------|------|
| `core/state.py` | `AgentState` 定义、节点名常量、相似度阈值 |
| `core/visualization.py` | 从 topology_data 生成 G6 交互拓扑 HTML |
| `agents/retriever.py` | 多引擎以图搜图，收集候选 URL |
| `agents/validator.py` | pHash + CLIP + OCR + 多模态 LLM 校验，去重合并 |
| `agents/analyzer.py` | Firecrawl/llmscrapy 爬取 + LLM 提取时间/互动量/传播角色 |
| `agents/orchestrator.py` | 流程编排、报告生成、数据持久化 |
| `tools/llmscrapy/` | 通用网页抓取管线，支持 Direct/Jina/Firecrawl 三种 fetcher |

## 工作流

```
upload → retrieve → validate → analyze → report
```

1. **上传图片** — 保存上传文件，提取图片元信息
2. **以图搜图** — 调用百度/Yandex/Google/Tineye/SauceNAO/SerpApi/ASCII2D 等多引擎搜索
3. **相似度校验** — pHash 快速筛 → CLIP 语义验证 → OCR 文字比对 → 多模态 LLM 复核 → 去重合并
4. **传播分析** — Firecrawl/llmscrapy 爬取网页 → LLM 提取发布时间/作者/互动量 → 传播关系图构建
5. **生成报告** — Markdown 报告 + G6 交互拓扑图 + 结构化数据下载

## Quickstart

### 1. 创建 Python 环境

```bash
conda create -n image_agent python=3.10
conda activate image_agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如需启用 CLIP 视觉语义：

```bash
pip install torch torchvision transformers
```

如需启用 OCR：

```bash
pip install rapidocr-onnxruntime paddleocr paddlepaddle
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API 密钥
```

必需的最小配置：
- `LLM_API_KEY` — DeepSeek / OpenAI 兼容 API 密钥
- `FIRECRAWL_API_KEY` — Firecrawl 网页抓取密钥（格式 `fc-xxx`）
- `SERPAPI_API_KEY` — Google Lens 搜图（可选但推荐）

### 4. 启动服务

```bash
python app.py
```

访问 `http://127.0.0.1:8502`

测试模式（加载已有报告，无需上传图片）：

```bash
python app.py --test
```

## 使用流程

1. 打开页面，上传一张待溯源图片
2. 点击"开始溯源"
3. 等待流程完成（页面实时显示进度和关键日志）
4. 查看结果：报告摘要、G6 拓扑图、分平台检索结果、原始数据

## 核心特性

### 多引擎以图搜图

支持百度、Yandex、Google、Tineye、SauceNAO、ASCII2D、SerpAPI Lens、Mitmproxy 代理八大引擎，可通过 `SEARCH_ENGINE` 环境变量配置。

### 三级视觉校验

- **pHash** — 快速全量筛选，强弱双阈值
- **CLIP** — 语义级图像+文本相似度
- **OCR** — 多引擎链文字提取，水印检测（微博/小红书/抖音平台识别）
- **多模态 LLM** — 边界案例复核

### 智能去重

强确定去重（同 URL / 同图片地址）直接合并；疑似重复（同站点同标题 / 图文相似 / 同图不同字）仅标记保留，避免误吞关键传播节点。

### llmscrapy 网页抓取

通用网页抓取管线，三种 fetcher 可选：
- `direct` — 直接 HTTP + 反爬策略（免费）
- `jina` — Jina AI Reader API（免费额度）
- `firecrawl` — Firecrawl API（付费，JS 渲染 + 住宅代理）

### 主流平台 API

- **微博** — Tikomni API 获取博文详情、互动量
- **小红书** — Tikomni API 获取笔记详情
- **百度百家号** — 直接解析 JSON-LD + 互动量 API

## Validator 配置说明

### 相似度阈值

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SIMILARITY_THRESHOLD` | 0.90 | 综合通过阈值 |
| `VALIDATOR_HASH_STRONG_PASS_THRESHOLD` | 0.92 | pHash 强通过 |
| `VALIDATOR_HASH_WEAK_PASS_THRESHOLD` | 0.72 | pHash 弱通过（需 CLIP 辅助） |
| `VALIDATOR_CLIP_REVIEW_THRESHOLD` | 0.78 | CLIP 复核阈值 |
| `VALIDATOR_LLM_BOUNDARY_SIMILARITY_FLOOR` | 0.75 | LLM 复核最低分 |
| `VALIDATOR_JOINT_DEDUP_THRESHOLD` | 0.86 | 联合去重阈值 |

### OCR 配置

```bash
VALIDATOR_ENABLE_OCR=true                      # 启用 OCR
VALIDATOR_ENABLE_OCR_PREFILTER=true            # 预筛减少 OCR 调用
VALIDATOR_OCR_PROVIDER=rapidocr,paddleocr_vl,baidu  # 引擎链
VALIDATOR_OCR_FALLBACK_ON_EMPTY=false          # 空结果不重试下一引擎
```

### CLIP 配置

```bash
VALIDATOR_ENABLE_CLIP=true                     # 启用 CLIP
VALIDATOR_CLIP_MODEL=openai/clip-vit-base-patch32
VALIDATOR_CLIP_LOCAL_ONLY=true                 # 仅使用本地模型
HF_ENDPOINT=https://hf-mirror.com              # HuggingFace 镜像
```

## 输出文件

每次运行在 `output/report_YYYYMMDD_HHMMSS/` 生成：

| 文件 | 内容 |
|------|------|
| `summary.json` | 报告摘要、分析汇总、拓扑数据 |
| `nodes_data.json` | 节点结构化数据（含相似度/互动量/传播角色） |
| `state_dump.json` | 完整 AgentState |
| `topology.html` | G6 交互拓扑图（可浏览器打开） |
| `logs.txt` | 关键里程碑日志 |
| `full_logs.txt` | 完整细粒度日志 |

## 本地调试

```bash
# CLI 运行完整工作流
python workflow.py
```

```bash
# 单独调试各 Agent
python agents/retriever.py
python agents/validator.py
python agents/analyzer.py
python agents/orchestrator.py
```

```bash
# llmscrapy 独立测试
python -m tools.llmscrapy.cli --url https://example.com --fetcher firecrawl
```

## 停止服务

在终端中按 `Ctrl+C`。
