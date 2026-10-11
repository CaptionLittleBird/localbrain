# localbrain

个人本地知识库 + Agent 问答。文档解析、向量化、重排、检索都在本地完成，只有生成回答时调用云端大模型（默认 DeepSeek）。

```
  浏览器 / 手机
       │  局域网 或 Tailscale
       ▼
  Open WebUI                 [Docker]
       │  OpenAI 兼容接口
       ▼
  localbrain serve           [原生]  FastAPI + LangGraph ───► DeepSeek API
       │                │
       ▼                ▼
  Qdrant [Docker]   oMLX [原生，用 GPU]  bge-m3 向量 + bge-reranker 重排
       ▲                ▲
       │                │
  localbrain watch [原生]   监听 data/docs → Docling 解析 → 切块 → 向量化 → 写入
```

- **解析**：用 Docling 转成 Markdown，保留标题层级和表格，PDF 按页、演示文稿按幻灯片记录页码；TXT、MD 直接读取（自动识别 UTF-8 / GBK）。支持的格式见下面「文档格式」。
- **入库**：解析在独立的子进程里并行进行，和向量化同时跑；单个文件卡住或崩溃只影响它自己（超时后终止，记为失败）。
- **检索**：向量 + jieba 分词的 BM25 做混合召回，再重排。向量和重排模型由 oMLX 提供，本项目通过 `/v1/embeddings`、`/v1/rerank` 调用。
- **Agent**：有三个工具：`search_kb` 检索、`read_document` 读上下文、`list_documents` 列文档。由模型决定查几次、查什么，回答时标注出处。
- **为什么模型不放在 Docker 里**：Mac 上的 Docker 跑在 Linux 虚拟机里，用不了 GPU（Metal）。oMLX 原生运行，通过 MLX 使用 GPU。

### 模型后端

向量模型和重排模型都可以在 `.env` 里切换后端（`EMBED_BACKEND` / `RERANK_BACKEND`）：

| 后端 | 说明 | 适用场景 |
|---|---|---|
| `omlx` | 调用 oMLX 的接口（也兼容 Xinference、硅基流动等同格式的服务） | Mac mini 上正式使用 |
| `local` | 本进程用 sentence-transformers 加载，模型名填 HuggingFace ID | 不装 oMLX 时的备选 |
| `none` | 不用模型：没有向量就只做 BM25 关键词检索，没有重排就用召回顺序 | Windows 上开发，不下载任何模型 |

PDF 解析同理：`PDF_PARSER=docling` 能识别标题和表格，但需要 Docling 的版面模型；`PDF_PARSER=simple` 只提取文字，不需要模型。

## 目录

```
src/localbrain/
  config.py      配置（读 .env）
  parsing.py     文档解析
  chunking.py    按标题结构切块
  models.py      向量模型、重排模型（omlx / local / none 三种后端）
  sparse.py      BM25 稀疏向量
  store.py       SQLite 文档清单 + Qdrant 向量库
  ingest.py      入库与目录监听
  workers.py     解析进程池（并行、超时、崩溃隔离）
  retrieval.py   混合检索 + 重排
  agent.py       LangGraph Agent 和工具
  server.py      OpenAI 兼容接口
  web.py         文档管理页面的接口（页面本身是 kb.html）
  citations.py   把回答里的出处换成打开原文的链接
  history.py     多轮对话时接回上一轮的检索结果
  cli.py         命令行
scripts/macos/   launchd 服务安装脚本
eval/            检索评估问题集
tests/           单元测试
```

## uv 用法（对照 pip）

uv 用来替代 pip 加 venv，速度快很多。本项目用它主要是为了 `uv.lock`：这个文件锁定了所有依赖的精确版本，保证 Windows 和 Mac 装出来的版本一致。`uv.lock` 要提交到 git。

安装 uv：Windows 上 `pip install uv`；Mac 上 `brew install uv`（或者同样 `pip install uv`）。

| 想做的事 | pip 的做法 | uv 的做法 |
|---|---|---|
| 建虚拟环境并装好依赖 | `python -m venv .venv` + `pip install -r requirements.txt` | `uv sync` |
| 加一个依赖 | `pip install httpx`，再手动写进 requirements | `uv add httpx`（自动写进 pyproject.toml 和 uv.lock） |
| 删一个依赖 | `pip uninstall httpx` | `uv remove httpx` |
| 运行程序 | 先激活环境，再 `python xxx.py` | `uv run python xxx.py`、`uv run localbrain ...` |
| 升级某个依赖 | `pip install -U httpx` | `uv lock --upgrade-package httpx`，再 `uv sync` |
| 查看已装的包 | `pip list` | `uv pip list`，或用 `uv tree` 看依赖树 |
| 临时装个包试试 | `pip install xxx` | `uv pip install xxx`（不写进项目，下次 `uv sync` 会被清掉） |

- `uv sync` 会把本项目以可编辑模式（相当于 `pip install -e .`）装进 `.venv`，改了代码直接生效，不用重装。
- `uv run` 运行前会检查环境和 `uv.lock` 是否一致，不一致就先同步，所以不用手动激活环境。

**也可以像以前一样激活环境再用：**

```powershell
.venv\Scripts\activate          # Windows；Mac 上是 source .venv/bin/activate
localbrain search "你的问题"     # 激活后直接用命令，不用加 uv run
python                          # 用的就是项目环境里的 Python
```

> 注意：uv 建的 `.venv` 里**没有 pip**。激活环境后敲 `pip install`，实际调用的是系统全局的 pip，包会装到别的地方。在项目里装包请用 `uv add` 或 `uv pip install`。

**完全不想用 uv 也行**，纯 pip 一样能跑：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

代价是不读 `uv.lock`，Windows 和 Mac 上装到的版本可能不一样。

## 在 Windows 上开发

开发时不需要下载任何模型。

```powershell
uv sync
copy .env.example .env
```

编辑 `.env`，填入 `LLM_API_KEY`，再把下面几项改成开发模式：

```ini
EMBED_BACKEND=none
RERANK_BACKEND=none
PDF_PARSER=simple
# 不想开 Docker 就用 Qdrant 嵌入模式（同一时间只能一个进程访问，ingest 和 serve 不能同时开）
QDRANT_URL=data/qdrant_local
```

想用 Docker 版的 Qdrant 就不加最后一行，改为启动 Docker Desktop 后运行 `docker compose up -d qdrant`。

```powershell
# 把几份文档放进 data/docs，然后：
uv run localbrain ingest              # 入库
uv run localbrain search "你的问题"    # 只看检索结果，不调大模型
uv run localbrain ask "你的问题"       # 完整问答
```

以上命令也可以写成 `python -m localbrain ...`（先激活 `.venv`）。

想在网页里用：先运行 `uv run localbrain serve`，再运行 `docker compose up -d`，然后打开 http://localhost:3000 。

如果局域网里的 Mac 上已经跑着 oMLX，Windows 也可以直接用它的模型：把 `EMBED_BACKEND`、`RERANK_BACKEND` 设成 `omlx`，`OMLX_BASE_URL` 填 `http://<Mac 的 IP>:8000/v1`。oMLX 监听局域网地址时要求设置 API key，`OMLX_API_KEY` 要和它保持一致。

> 从 `none` 切换到 `omlx`（或者换了向量模型）后，要先运行 `uv run localbrain reset --yes` 再重新入库，程序检测到不一致时也会提示。

PyPI 依赖走清华镜像（见 `pyproject.toml`）。

## 部署到 Mac mini

**1. 系统准备**

```bash
brew install uv git
brew install --cask orbstack       # 比 Docker Desktop 省内存，用法一样
```

oMLX 按它的官方说明安装（macOS App 或 Homebrew），并设为登录后自动启动。

- 系统设置 → 能耗：打开"防止自动睡眠"和"断电后自动启动"。
- 系统设置 → 用户与群组：打开自动登录。OrbStack 和 launchd 用户服务都要在用户登录后才会运行。注意开着 FileVault 时无法自动登录。

**2. 在 oMLX 里准备模型**

在 oMLX 管理后台（默认 http://localhost:8000/admin）下载向量模型和重排模型的 MLX 版本：bge-m3，以及 bge-reranker-v2-m3 或其他 XLM-RoBERTa 架构的重排模型。然后确认 oMLX 对外暴露的模型名：

```bash
curl http://127.0.0.1:8000/v1/models
```

把列出来的名字分别填到 `.env` 的 `EMBED_MODEL`、`RERANK_MODEL`（默认值是 `bge-m3` 和 `bge-reranker-v2-m3`）。oMLX 只监听本机时可以不设 API key；如果改成监听局域网，oMLX 要求必须设 key，`OMLX_API_KEY` 要和它保持一致。

**3. 拉代码并配置**

```bash
git clone <你的仓库> ~/localbrain && cd ~/localbrain
uv sync
cp .env.example .env && vi .env    # 填 LLM_API_KEY、LOCALBRAIN_API_KEY、WEBUI_SECRET_KEY、WEB_PASSWORD，核对模型名
docker compose up -d
```

默认端口：oMLX 8000、本项目 API 8800、Qdrant 6333、Open WebUI 3000。被占用的话，改 `.env` 里的 `API_PORT`、`QDRANT_PORT`、`WEBUI_PORT`；oMLX 换了端口就改 `OMLX_BASE_URL`。docker compose 和 Python 程序都从 `.env` 读取这些配置，改完后重新运行 `docker compose up -d`，再重启 launchd 服务（见第 5 步）。查看端口占用：`lsof -i :8800`。

> 项目目录和文档目录不要放在"文稿"、"桌面"、"下载"或 iCloud 里：macOS 的隐私保护会拦截 launchd 后台服务访问这些位置。

**4. 首次入库**

```bash
uv run localbrain ingest
uv run localbrain search "随便问一个文档里有的问题"
```

日志开头会显示 `向量模型：bge-m3（oMLX http://127.0.0.1:8000/v1）`。连不上 oMLX 或模型名不对时，会直接提示原因。

`PDF_PARSER=docling` 在第一次解析 PDF 时会自动从 HuggingFace 下载 Docling 的版面模型（约 370MB），国内可能很慢，可以按下面的「Docling 模型」一节预先下载。

**5. 注册成开机自启的服务**

```bash
bash scripts/macos/install-services.sh
```

不要加 sudo，这是当前用户的登录服务。脚本会注册 `com.localbrain.serve`（API）和 `com.localbrain.watch`（监听文档目录、自动入库）两个服务，进程挂了会自动重启。日志在 `~/Library/Logs/localbrain/`。修改 `.env` 后重新执行这个脚本即可重启服务。

**6. 使用**

浏览器打开 `http://<Mac 的名字>.local:3000`（端口以 `WEBUI_PORT` 为准）。第一个注册的账号就是管理员，建好之后在管理员设置里关掉新用户注册。在外面用手机访问的话，Mac 和手机都装上 Tailscale，然后用 Tailscale 分配的地址访问。

文档管理页面在 `http://127.0.0.1:8800/kb`（端口以 `API_PORT` 为准），用 `WEB_PASSWORD` 登录。页面按文件夹管理文档：上传文件或整个文件夹、新建文件夹、移动、重命名、删除，查看入库状态和失败原因，在线打开原文。回答里的出处会自动变成链接，点开直接看原文（没登录的话先登录，登录后自动跳回去）：PDF 的 `[文件名 p.12]` 跳到对应页；EPUB、Word 等没有页码的格式写成 `[文件名 § 章节]`，打开阅读页定位到那一节，左边有章节目录可以前后翻。多轮对话时，上一轮查到的内容会接回给模型，追问不用从头检索。想在其他设备上用，把 `API_HOST` 改成 `0.0.0.0`，这时 `LOCALBRAIN_API_KEY` 也会暴露在局域网上，务必换成随机字符串（换了之后要在 Open WebUI 管理员设置 → 外部连接里同步修改）。出处链接的地址默认是 `http://<本机名>.local:8800`，用 Tailscale 在外面访问时把 `PUBLIC_URL` 设成 Tailscale 的地址。

## 日常使用

| 做什么 | 怎么做 |
|---|---|
| 加文档 | 在文档管理页面上传（可以整个文件夹拖进去），或者直接拷进 `data/docs`，watch 服务会自动入库；删除文件后也会同步移除 |
| 整理文档 | 在文档管理页面新建文件夹、移动、重命名（可以多选，也可以把文件拖到左边的文件夹里），或者直接在 Finder 里整理。内容没变的文件移动、改名后只更新路径，不会重新入库 |
| 看入库情况 | 文档管理页面，或 `uv run localbrain status`（失败的文档会显示原因） |
| 重试失败的文档 | 页面上点「重试」，或 `uv run localbrain ingest --force` |
| 重新入库 | `uv run localbrain ingest --force` |
| 调检索效果 | `uv run localbrain search "问题" -k 10` |
| 评估命中率 | `uv run localbrain eval eval/questions.jsonl`。每行一个问题，`expected` 是期望命中的文件名关键字，可选的 `pages` 是答案所在页码范围，精确到页判断命中。改检索参数、换模型前后各跑一次对比 |
| 换向量模型或 `EMBED_BACKEND` | 改完后执行 `uv run localbrain reset --yes`，再重新入库 |
| 看日志 | `tail -f ~/Library/Logs/localbrain/serve.log` |

## 文档格式

| 格式 | 说明 |
|---|---|
| PDF | 有文字层的直接提取；缺文字层的页超过两成就按扫描件 OCR（`OCR=auto`）。会按 PDF 书签和章节编号推断标题层级 |
| EPUB | 按书的阅读顺序解析，保留章节标题和表格。有 DRM 的书读不到正文，会提示需要先去掉 DRM |
| Word：.docx .docm、.doc .rtf | .doc、.rtf 在 macOS 上用系统自带的 textutil 转换，不用装 LibreOffice |
| 表格：.xlsx .xlsm .csv、演示文稿：.pptx .ppsx .pptm | 演示文稿按幻灯片标注页码 |
| OpenDocument：.odt .ods .odp；iWork：.pages .key .numbers | |
| 网页和邮件：.html .htm .xhtml .mhtml .mht、.eml .msg | |
| 其他文本：.txt .md .markdown .adoc .asciidoc .tex | |
| 图片：.png .jpg .jpeg .tif .tiff .bmp .webp | OCR 识别文字，只在 macOS 上可用 |
| .xls .ppt（老版 Office） | 需要安装 LibreOffice：`brew install --cask libreoffice`。没装的话入库会失败并提示原因 |

内容完全相同的文件（比如同一本书拷了两份）只入库一份，另一份在文档管理页面显示为「重复」；原件删除后会自动改用它。同一本书的 PDF 和 EPUB 都入库的话，检索结果会去掉内容几乎一样的片段，不会重复占名额。

MOBI、AZW3 等 Kindle 格式不支持，可以用 Calibre 批量转成 EPUB：`ebook-convert 书名.azw3 书名.epub`。

**性能参考（M4 Mac mini，1051 页的《鸟哥的 Linux 私房菜》）**：解析 4 分多钟（约 0.25 秒/页；扫描件要 OCR，约 0.6 秒/页），切成约 2100 段，向量化 5 分钟左右，整本约 10 分钟。向量化是最慢的一步，受 GPU 算力限制，加大批量、并发都不会更快；多个文件时，解析和向量化会同时进行。

## Docling 模型（可选，预先下载）

`PDF_PARSER=docling` 需要两个模型：版面分析模型和表格识别模型。HuggingFace 在国内很慢，hf-mirror 对这类大文件也只是转发回 HuggingFace，起不到加速作用。ModelScope（魔搭）上有同样的模型，可以预先下载：

```bash
# 版面分析模型（约 164MB）
uv run modelscope download ds4sd/docling-layout-heron \
  --local-dir data/models/docling/docling-project--docling-layout-heron
# 表格识别模型，只下 accurate 版（约 203MB）
uv run modelscope download ds4sd/docling-models \
  --include "model_artifacts/tableformer/accurate/*" \
  --local-dir data/models/docling/docling-project--docling-models
```

然后在 `.env` 里设置 `DOCLING_ARTIFACTS_PATH=data/models/docling`。目录名必须和上面一致，Docling 按这个名字查找模型。

> 魔搭上这两个模型的权重文件和 HuggingFace 上 Docling 使用的版本 sha256 一致，是同一份模型。

## 开发

```bash
uv sync                  # 包含 pytest 等开发依赖
uv run pytest            # 单元测试，不需要模型和外部服务，十几秒跑完
uv run localbrain eval eval/questions.jsonl   # 检索效果评估，需要 oMLX 和 Qdrant
```

`eval/questions.jsonl` 是按《鸟哥的 Linux 私房菜》出的 68 个问题。当前结果：第 1 条命中 88%，前 3 条命中 96%，平均每次检索 2.3 秒。

## 备份

真正需要备份的只有 `data/docs`（原始文档）。向量库和 `data/localbrain.db` 都可以用 `localbrain reset --yes && localbrain ingest` 重建。Open WebUI 的聊天记录在 Docker 卷 `open_webui_data` 里，需要的话一起备份。

## 注意

- 检索到的文档片段会随问题发送给 DeepSeek，敏感文档请留意。
- Open WebUI 的环境变量只在第一次启动时生效，之后要改设置得去它的管理员界面。
- 目前按 `deepseek-chat`（非思考模式）设计。思考模式（`deepseek-reasoner`）的工具调用协议不同，还没有适配。
- 换成其他 OpenAI 兼容的模型（通义千问、Kimi、GLM 等）只需要改 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`。
