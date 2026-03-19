# AstralAI

AstralAI 是一个基于 FastAPI 的轻量级 AI 聊天后端，提供流式聊天、会话管理和 PostgreSQL 持久化能力。当前实现围绕 Anthropic 兼容模型接入，支持两类交互模式：

- `thinking_enabled=true`：直接进入主聊天流，返回回答分片与链式执行轨迹。
- `thinking_enabled=false`：先进行 simple / complex / agent 路由，再根据路由结果继续输出。

项目目标很明确：保持后端结构简单，先把可用的聊天 API、会话状态和调试链路搭起来。

## 配套前端

AstralAI 的前端项目位于：

- `astral-ai-web`：<https://github.com/Ayor1337/astral-ai-web>

典型的本地联调方式是先启动本仓库提供的后端 API，再单独启动前端项目，通过 HTTP 接口访问本服务。

## 功能特性

- 基于 SSE 的流式聊天接口 `POST /api/chat/stream`
- 支持显式创建空会话，也兼容首轮消息隐式建会话
- 支持通过 `run_id` 请求终止当前生成
- 支持会话列表、详情、标题修改和软删除
- 支持对话短期记忆窗口与摘要触发阈值配置
- 提供 `pytest` 自动化测试与 `test_main.http` 手动调试样例

## 技术栈

- Python 3.13
- FastAPI
- Uvicorn
- PostgreSQL
- Anthropic 兼容模型接口

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 启动 PostgreSQL

项目自带一个本地开发用的 `docker-compose.yml`：

```bash
docker compose up -d
```

默认数据库连接为：

```text
postgresql://postgres:postgres@localhost:5432/astral_ai
```

### 3. 配置环境变量

先复制一份环境变量模板：

```bash
copy .env.example .env
```

然后至少补齐以下配置：

```env
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=your-anthropic-model
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/astral_ai
```

如果你使用的是兼容 Anthropic API 的代理服务，也可以额外配置：

```env
ANTHROPIC_BASE_URL=https://your-provider.example.com/anthropic
TITLE_AGENT_MODEL=your-title-model
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload
```

启动后可访问：

- 根路径健康检查：`GET http://127.0.0.1:8000/`
- Swagger 文档：`http://127.0.0.1:8000/docs`
- ReDoc 文档：`http://127.0.0.1:8000/redoc`

### 5. 连接前端项目

后端启动后，可继续拉起前端仓库进行联调：

```text
https://github.com/Ayor1337/astral-ai-web
```

建议顺序：

- 先启动 PostgreSQL
- 再启动本仓库后端服务
- 最后启动前端项目并将其请求指向 `http://127.0.0.1:8000`

## 配置说明

`.env.example` 当前包含以下变量：

| 变量名 | 说明 |
| --- | --- |
| `ANTHROPIC_API_KEY` | 必填，上游模型服务密钥 |
| `ANTHROPIC_BASE_URL` | 可选，Anthropic 兼容服务的基础地址，必须以 `http://` 或 `https://` 开头 |
| `ANTHROPIC_MODEL` | 必填，主聊天模型名称 |
| `TITLE_AGENT_MODEL` | 可选，用于异步生成会话标题的模型名称 |
| `DATABASE_URL` | PostgreSQL 连接串，仅支持 `postgresql://` 或 `postgresql+asyncpg://` |
| `MEMORY_WINDOW_SIZE` | 短期记忆窗口大小，必须大于 0 |
| `MEMORY_SUMMARY_TRIGGER` | 触发摘要整理的阈值，必须大于 `MEMORY_WINDOW_SIZE` |

配置校验逻辑位于 `app/core/config.py`，启动或请求处理期间如果发现配置错误，会直接返回稳定的错误响应。

## API 概览

### 聊天接口

`POST /api/chat/stream`

请求体示例：

```json
{
  "conversation_id": null,
  "message": "查一下 207.97.137.107",
  "thinking_enabled": true
}
```

返回内容为 `text/event-stream`。常见事件包括：

- `conversation`：返回 `conversation_id`、`title`、`run_id`
- `chunk`：回答文本分片
- `route` / `planner_done`：仅在 `thinking_enabled=false` 且命中复杂路径时返回
- `thought_step` / `trace_step` / `trace_done`：链式执行轨迹
- `done`：流式输出结束

### 终止当前生成

`POST /api/chat/runs/{run_id}/stop`

`run_id` 来自聊天流首个 `conversation` 事件。

### 会话接口

- `POST /api/conversations`：显式创建空会话
- `GET /api/conversations`：获取会话列表
- `GET /api/conversations/{conversation_id}`：获取会话详情
- `PATCH /api/conversations/{conversation_id}`：更新会话标题
- `DELETE /api/conversations/{conversation_id}`：软删除会话

更完整的请求样例见 `test_main.http`。

## 项目结构

```text
app/
├─ api/            # HTTP 路由入口
├─ core/           # 配置加载与校验
├─ db/             # 数据库会话与模型
├─ llm/            # 模型调用与代理逻辑
├─ repositories/   # 数据访问层
├─ schemas/        # Pydantic 请求/响应模型
├─ services/       # 业务编排与记忆管理
└─ main.py         # FastAPI 应用入口

tests/
├─ test_chat_api.py
├─ test_chat_service.py
├─ test_config.py
├─ test_conversations_api.py
├─ test_llm_anthropic_compat.py
└─ test_memory_service.py
```

## 测试

运行全部测试：

```bash
pytest
```

只跑聊天接口测试：

```bash
pytest tests/test_chat_api.py
```

## 手动调试

仓库根目录提供 `test_main.http`，适合在 JetBrains HTTP Client 或 VS Code REST Client 中直接调试：

- 健康检查
- 显式新建空会话
- 流式聊天
- 复杂路由命中
- 终止当前生成
- 会话 CRUD
- 直连上游模型复现 planner 请求

## 当前限制

- 当前后端默认允许任意来源跨域，适合开发阶段，不适合直接用于生产环境
- 数据库未配置时，应用会跳过数据库初始化，但涉及会话持久化的接口会返回配置错误
- README 目前聚焦本地开发与接口说明，未覆盖部署、监控和生产安全配置
