# AstralAI

AstralAI 是一个基于 FastAPI 的轻量级 AI 聊天后端，提供流式聊天、会话管理和 PostgreSQL 持久化能力。当前只保留两种交互模式：

- `thinking_enabled=false`：直接返回正文分片
- `thinking_enabled=true`：返回正文分片，同时把 thinking、tool_call、tool_result、search、fetch 等过程统一放进 `trace_step`

## 配套前端

- `astral-ai-web`：<https://github.com/Ayor1337/astral-ai-web>

## 功能特性

- 基于 SSE 的流式聊天接口 `POST /api/chat/stream`
- 支持显式创建空会话，也兼容首轮消息隐式建会话
- 支持通过 `run_id` 请求终止当前生成
- 支持会话列表、详情、标题修改和软删除
- 支持对话短期记忆窗口与摘要触发阈值配置

## 技术栈

- Python 3.13
- FastAPI
- Uvicorn
- PostgreSQL
- `anthropic` / `openai`

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 启动 PostgreSQL

```bash
docker compose up -d
```

默认数据库连接：

```text
postgresql://postgres:postgres@localhost:5432/astral_ai
```

### 3. 配置环境变量

```bash
copy .env.example .env
```

至少补齐：

```env
LLM_PROVIDER=anthropic
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/astral_ai
```

如需代理地址，可额外配置：

```env
LLM_BASE_URL=https://your-provider.example.com
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload
```

## 配置说明

| 变量名 | 说明 |
| --- | --- |
| `LLM_PROVIDER` | 必填，默认聊天 provider，当前仅支持 `anthropic` 与 `openai` |
| `LLM_API_KEY` | 必填，默认聊天 provider 的上游密钥 |
| `LLM_BASE_URL` | 可选，默认聊天 provider 的基础地址，必须以 `http://` 或 `https://` 开头 |
| `LLM_MODEL` | 必填，默认聊天模型名称 |
| `DATABASE_URL` | PostgreSQL 连接串，仅支持 `postgresql://` 或 `postgresql+asyncpg://` |
| `MEMORY_WINDOW_SIZE` | 短期记忆窗口大小，必须大于 0 |
| `MEMORY_SUMMARY_TRIGGER` | 触发摘要整理的阈值，必须大于 `MEMORY_WINDOW_SIZE` |

需要注意：

- `thinking_enabled=true` 目前只支持 `anthropic` provider
- 当默认聊天 provider 为 `openai` 且请求开启 `thinking_enabled` 时，接口会直接返回 400

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
- `chunk`：正文文本分片
- `trace_step` / `trace_done`：仅在 `thinking_enabled=true` 时返回
- `done`：流式输出结束

`trace_step.type` 当前可包含：

- `thinking`
- `tool_call`
- `tool_result`
- `search`
- `fetch`
- `retry`
- `other`

其中 `thinking` 节点会复用同一个 `step_id`：

- 思考过程中持续返回 `trace_step(status="running")`
- 思考结束时，对同一 `step_id` 再回写一条 `trace_step(status="success")`

前端应按 `step_id` 做 upsert，而不是把这两条当成两个独立节点。

### 终止当前生成

`POST /api/chat/runs/{run_id}/stop`

`run_id` 来自聊天流首个 `conversation` 事件。

### 会话接口

- `POST /api/conversations`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}`
- `PATCH /api/conversations/{conversation_id}`
- `DELETE /api/conversations/{conversation_id}`

会话详情中的 assistant 消息只保留：

- `content`
- `trace_steps`

## 测试

运行全部测试：

```bash
pytest
```

只跑聊天相关测试：

```bash
pytest tests/test_chat_api.py tests/test_chat_service.py
```

## 手动调试

仓库根目录提供 `test_main.http`，适合在 JetBrains HTTP Client 或 VS Code REST Client 中直接调试。
