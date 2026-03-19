# AstralAI API 文档

## 概览

- 基础地址：`http://127.0.0.1:8000`
- Swagger：`/docs`
- ReDoc：`/redoc`
- 响应编码：`UTF-8`

当前后端提供以下能力：
- `GET /`
- `POST /api/chat/stream`
- `POST /api/chat/runs/{run_id}/stop`
- `POST /api/conversations`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}`
- `PATCH /api/conversations/{conversation_id}`
- `DELETE /api/conversations/{conversation_id}`

服务端会持久化会话历史、消息记录、摘要记忆和结构化轨迹信息。

## 运行前配置

服务启动前需要在项目根目录配置环境变量或 `.env` 文件。

```env
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_BASE_URL=
ANTHROPIC_MODEL=your-anthropic-model
TITLE_AGENT_MODEL=
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/astral_ai
MEMORY_WINDOW_SIZE=8
MEMORY_SUMMARY_TRIGGER=12
```

说明：
- `ANTHROPIC_API_KEY`：Anthropic API Key，必填。
- `ANTHROPIC_BASE_URL`：Anthropic 自定义服务地址，可选；为空时使用官方默认地址。
- `ANTHROPIC_MODEL`：Anthropic 主聊天模型名称，必填。
- planner 路由模型当前固定为 `MiniMax-M2`。
- `TITLE_AGENT_MODEL`：标题代理使用的可选模型名称；为空时回退 `ANTHROPIC_MODEL`。
- `DATABASE_URL`：PostgreSQL 连接地址，必填。
- `MEMORY_WINDOW_SIZE`：短期记忆窗口大小，默认 `8`。
- `MEMORY_SUMMARY_TRIGGER`：触发摘要压缩的消息数阈值，默认 `12`，必须大于 `MEMORY_WINDOW_SIZE`。

## 会话接口

### 新建空会话

```http
POST /api/conversations
```

成功响应：

```json
{
  "id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "title": "新对话",
  "summary": null,
  "created_at": "2026-03-17T10:00:00Z",
  "updated_at": "2026-03-17T10:00:00Z"
}
```

### 查询会话详情

```http
GET /api/conversations/{conversation_id}
```

成功响应示例：

```json
{
  "id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "title": "IP 查询排查",
  "summary": "用户正在排查一个 IP 地址的归属与来源。",
  "created_at": "2026-03-17T10:00:00Z",
  "updated_at": "2026-03-17T10:05:00Z",
  "messages": [
    {
      "role": "user",
      "content": "查一下 207.97.137.107",
      "reasoning_summary": null,
      "trace_steps": null,
      "sequence": 1,
      "created_at": "2026-03-17T10:00:00Z"
    },
    {
      "role": "assistant",
      "content": "我先查一下这个 IP 的信息。",
      "reasoning_summary": null,
      "trace_steps": [
        {
          "step_id": "search-1",
          "parent_step_id": "assistant-thinking-0f31-2",
          "type": "search",
          "kind": "result_list",
          "status": "success",
          "title": "搜索 IP 信息",
          "message": "先搜索可用的 IP 查询站点。",
          "query": "207.97.137.107 IP lookup",
          "result_count": 2,
          "timestamp": "2026-03-18T12:00:01+00:00",
          "order": 1,
          "payload": {
            "items": [
              {
                "title": "IP Address Lookup",
                "url": "https://example.com/ip",
                "domain": "example.com",
                "snippet": "Lookup an IP"
              }
            ]
          }
        }
      ],
      "sequence": 2,
      "created_at": "2026-03-17T10:00:03Z"
    }
  ]
}
```

说明：
- `reasoning_summary` 为兼容旧字段，当前不承担思考链展示职责。
- `trace_steps` 为历史回放使用的链式轨迹数据，其中包含步骤化的 `thought` 节点。
- 前端实时显示链时，应同时兼容 `thought_step` 和 `trace_step`。

## 流式聊天

### 请求

```http
POST /api/chat/stream
Accept: text/event-stream
Content-Type: application/json
```

说明：
- 首个 `conversation` 事件会返回 `run_id`。
- 前端点击“终止按钮”时，应调用 `POST /api/chat/runs/{run_id}/stop`。
- 停止请求成功后，继续监听当前 SSE，直到收到 `done.status=stopped`。

### 请求体

```json
{
  "conversation_id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "message": "查一下 207.97.137.107",
  "thinking_enabled": false
}
```

字段说明：
- `conversation_id`
  - 类型：`string(UUID)`
  - 必填：否
  - 不传时，服务端会隐式新建会话。
- `message`
  - 类型：`string`
  - 必填：是
  - 最少 1 个字符。
- `thinking_enabled`
  - 类型：`boolean`
  - 必填：否
  - 默认：`false`
  - `false`：先走路由，再决定 simple 还是 complex/agent 回复。
  - `true`：跳过路由，直接进入复杂执行路径。

## 终止当前生成

### 请求

```http
POST /api/chat/runs/{run_id}/stop
```

### 成功响应

```json
{
  "run_id": "6e0f1938-897a-4dda-b17c-1c33d7ef8d24",
  "status": "stop_requested"
}
```

说明：
- `run_id` 来自当前 SSE 首个 `conversation` 事件。
- 若 `run_id` 不存在，或该轮生成已经结束，接口返回 `404`。
- stop 接口成功后，当前 SSE 会以 `done.status=stopped` 收尾。

## SSE 时序

### 模式 A：`thinking_enabled=false` 且命中 `simple`

```text
conversation -> chunk* -> done
```

示例：

```text
event: conversation
data: {"conversation_id":"0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98","title":"新对话","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}

event: chunk
data: {"content":"你好！"}

event: chunk
data: {"content":" 我在。"}

event: done
data: {"status":"completed","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}
```

说明：
- simple 路径不会对前端暴露 `route`、`planner_done`。
- simple 路径不会返回 `trace_step`。
- 前端只会显示普通回复，不显示链。
- 首轮标题会在后台异步生成并落库；当前 SSE 不保证返回标题更新事件。

### 模式 B：`thinking_enabled=false` 且命中 `complex/agent`

```text
conversation -> route -> planner_done -> trace_step* + chunk* -> trace_done -> done
```

示例：

```text
event: conversation
data: {"conversation_id":"0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98","title":"新对话","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}

event: route
data: {"route":"agent","plan":["搜索相关信息","抓取候选链接"],"tools":["web_search","http_fetch"]}

event: planner_done
data: {"status":"completed"}

event: trace_step
data: {"step_id":"search-1","type":"search","kind":"result_list","status":"success","message":"已搜索到候选结果","timestamp":"2026-03-18T12:00:00+00:00","order":1}

event: chunk
data: {"content":"我先查一下这个 IP。"}

event: trace_done
data: {"status":"completed"}

event: done
data: {"status":"completed","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}
```

说明：
- 只有最终进入复杂执行路径时，前端才会同时收到 `route/planner_done` 和链式 `trace_step`。
- 只要收到 `trace_step`，前端就应该显示链。
- 首轮标题会在后台异步生成并落库；当前 SSE 不保证返回标题更新事件。

### 模式 C：`thinking_enabled=true`

```text
conversation -> thought_step* -> trace_step* -> chunk* -> trace_done -> done
```

示例：

```text
event: conversation
data: {"conversation_id":"0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98","title":"新对话","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}

event: thought_step
data: {"step_id":"assistant-thought-0f31-2-1","type":"thought","status":"running","title":"确定查询方向","message":"先搜索可用的 IP 信息来源。","timestamp":"2026-03-18T12:00:00+00:00","order":-1000}

event: thought_step
data: {"step_id":"assistant-thought-0f31-2-1","type":"thought","status":"success","title":"确定查询方向","message":"先搜索可用的 IP 信息来源。","timestamp":"2026-03-18T12:00:01+00:00","order":-1000}

event: thought_step
data: {"step_id":"assistant-thought-0f31-2-2","type":"thought","status":"running","title":"准备整理结果","message":"准备汇总搜索和抓取结果后回答用户。","timestamp":"2026-03-18T12:00:01+00:00","order":-999}

event: trace_step
data: {"step_id":"search-1","parent_step_id":"assistant-thought-0f31-2-2","type":"search","kind":"result_list","status":"success","title":"搜索 IP 信息","message":"先搜索可用的 IP 查询站点。","query":"207.97.137.107 IP lookup","result_count":2,"timestamp":"2026-03-18T12:00:01+00:00","order":1}

event: chunk
data: {"content":"我先查一下这个 IP 的信息。"}

event: trace_done
data: {"status":"completed"}

event: done
data: {"status":"completed","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}
```

说明：
- `thinking_enabled=true` 时不会发 `route/planner_done`。
- 这条路径始终视为复杂执行路径。
- 服务端会优先发送 `thought_step`，首个 `chunk` 不会早于首个 `thought_step`。
- 若思考步骤改写失败或超时，会回退为旧的单节点 `trace_step(type=thought)`。
- 首轮标题会在后台异步生成并落库；当前 SSE 不保证返回标题更新事件。

### 模式 D：主动终止

```text
conversation -> chunk* / trace_step* -> POST /api/chat/runs/{run_id}/stop -> trace_done? -> done
```

示例：

```text
event: conversation
data: {"conversation_id":"0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98","title":"新对话","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}

event: chunk
data: {"content":"我先查到一部分结果。"}

event: done
data: {"status":"stopped","run_id":"6e0f1938-897a-4dda-b17c-1c33d7ef8d24"}
```

说明：
- 若当前轮已经产生链式轨迹，服务端会先发送 `trace_done {"status":"stopped"}`，再发送 `done {"status":"stopped","run_id":"..."}`。
- 服务端会保留本轮已生成的部分文本与已累计的 `trace_steps`。
- 若终止发生在首个 assistant 输出前，则不会新增 assistant 消息。

## SSE 事件说明

`conversation`
- 当前消息归属的会话。
- `data.conversation_id` 为会话 ID。
- `data.title` 为当前会话标题。
- `data.run_id` 为当前这一次生成的唯一标识。

`route`
- 仅在 `thinking_enabled=false` 且最终命中 `complex/agent` 时出现。
- `data.route` 可取：`complex`、`agent`。
- `data.plan` 仅在 `complex/agent` 时出现。
- `data.tools` 仅在 `agent` 时出现。

`planner_done`
- 仅在 `thinking_enabled=false` 且最终命中 `complex/agent` 时出现。
- 表示路由阶段结束，后续会继续进入复杂回复。

`chunk`
- assistant 文本输出分片。
- `data.content` 为当前分片文本。

`thought_step`
- 仅在 `thinking_enabled=true` 且思考步骤整理成功时出现。
- 表示面向前端展示的思考步骤，支持按 `step_id` upsert 更新。
- 常用字段：
  - `step_id`
  - `type`
  - `status`
  - `title`
  - `message`
  - `timestamp`
  - `order`
- `type` 固定为 `thought`。
- `status` 常见取值：
  - `running`：当前步骤正在展示中
  - `success`：该步骤已完成
  - `skipped`：当前轮被手动终止
- 生命周期规则：
  - 一个新步骤首次出现时，服务端会发送 `running`
  - 当下一步出现或当前轮结束时，上一条 `running` 步骤会被补发为 `success`
  - 前端应按 `step_id` 做增量更新，而不是简单 append

`trace_step`
- 链式执行轨迹节点，支持按 `step_id` upsert 更新。
- 常用字段：
  - `step_id`
  - `parent_step_id`
  - `type`
  - `kind`
  - `status`
  - `title`
  - `message`
  - `query`
  - `url`
  - `result_count`
  - `timestamp`
  - `order`
  - `retry_of`
  - `payload`
- `type` 可取：`thought/search/fetch/tool_call/tool_result/retry/other`。
- `status` 可取：`pending/running/success/error/skipped`。
- 当 `thinking_enabled=true` 时，`trace_step` 主要承载工具和执行轨迹；思考步骤优先通过 `thought_step` 返回。
- 若 `thought_step` 生成失败或超时，服务端会回退为单节点 `trace_step(type=thought)`。

`trace_done`
- 当前轮链式轨迹输出完成。
- `data.status` 可取：`completed`、`stopped`。

`done`
- 整个 SSE 请求结束。
- `data.status` 可取：`completed`、`stopped`。
- `data.run_id` 为当前轮生成 ID。
- 当前 SSE 收尾不会等待自动标题生成；最新标题应通过后续会话列表或详情接口刷新获取。

`error`
- 流式过程中出现异常。
- 若出现 `error`，本次流会提前终止，后续不会再发送 `done`。

## 错误响应

### 422 请求参数错误

触发场景：
- `message` 缺失或为空
- `conversation_id` 不是合法 UUID

### 404 会话不存在

```json
{
  "detail": "conversation not found"
}
```

### 404 聊天运行不存在

```json
{
  "detail": "chat run not found"
}
```

### 500 服务配置错误

```json
{
  "detail": "ANTHROPIC_BASE_URL must start with http:// or https://"
}
```

### 502 上游模型服务错误

```json
{
  "detail": "model upstream failed"
}
```

说明：
- 若异常发生在首个 SSE 事件之前，HTTP 状态码为 `502`。
- 若异常发生在 SSE 已开始之后，HTTP 状态码仍为 `200`，服务端会改为发送 `event: error`。

## 调试示例

### 使用 HTTP Client

仓库已提供示例文件 [test_main.http](/C:/Users/ayor/PycharmProjects/AstralAI/test_main.http)。

### 使用 curl

simple 路径示例：

```bash
curl -N http://127.0.0.1:8000/api/chat/stream \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"你好\",\"thinking_enabled\":false}"
```

complex/agent 路径示例：

```bash
curl -N http://127.0.0.1:8000/api/chat/stream \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"帮我查一下这个 IP 的归属和来源\",\"thinking_enabled\":false}"
```

复杂链直达示例：

```bash
curl -N http://127.0.0.1:8000/api/chat/stream \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"查一下 207.97.137.107\",\"thinking_enabled\":true}"
```

终止示例：

1. 先发起 `/api/chat/stream`，从首个 `conversation` 事件读取 `run_id`。
2. 再请求：

```bash
curl -X POST http://127.0.0.1:8000/api/chat/runs/6e0f1938-897a-4dda-b17c-1c33d7ef8d24/stop
```

## 备注

- 前端是否显示链，应判断当前轮是否收到 `thought_step` 或 `trace_step`。
- simple 路径没有链，也不会暴露路由信息。
- complex/agent 路径才会同时暴露路由信息和链式轨迹。
- `reasoning_summary` 仅为兼容旧字段，当前不再承担思考链展示职责。
- “终止按钮”的推荐实现方式是：保存当前轮 `run_id`，点击按钮后调用 stop 接口，并等待 SSE 以 `done.status=stopped` 收尾。
