# AstralAI API 文档

## 概览

- 基础地址：`http://127.0.0.1:8000`
- Swagger：`/docs`
- ReDoc：`/redoc`

当前后端提供：

- `GET /`
- `POST /api/chat/stream`
- `POST /api/chat/runs/{run_id}/stop`
- `POST /api/conversations`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}`
- `PATCH /api/conversations/{conversation_id}`
- `DELETE /api/conversations/{conversation_id}`

## 环境变量

```env
LLM_PROVIDER=anthropic
LLM_API_KEY=your-api-key
LLM_BASE_URL=
LLM_MODEL=your-model-name
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/astral_ai
MEMORY_WINDOW_SIZE=8
MEMORY_SUMMARY_TRIGGER=12
```

说明：

- `thinking_enabled=true` 当前仅支持 `anthropic`
- `thinking_enabled=false` 不会暴露任何链路事件

## 流式聊天

### 请求

```http
POST /api/chat/stream
Accept: text/event-stream
Content-Type: application/json
```

### 请求体

```json
{
  "conversation_id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "message": "查一下 207.97.137.107",
  "thinking_enabled": true
}
```

字段说明：

- `conversation_id`：可选，不传时服务端会隐式建会话
- `message`：必填
- `thinking_enabled`
  - `false`：只返回正文
  - `true`：正文之外的过程节点统一走 `trace_step`

### SSE 时序

#### 非思考模式

```text
conversation -> chunk* -> done
```

#### 思考模式

```text
conversation -> trace_step* -> thinking(success upsert)? -> chunk* -> trace_done -> done
```

### SSE 事件

`conversation`

- 返回 `conversation_id`、`title`、`run_id`

`chunk`

- 返回正文文本分片

`trace_step`

- 仅在 `thinking_enabled=true` 时出现
- 统一承载所有非正文过程节点
- 前端应按 `step_id` 做 upsert 更新，而不是把同一 `step_id` 的多次事件当成多个节点
- 常见 `type`：
  - `thinking`
  - `tool_call`
  - `tool_result`
  - `search`
  - `fetch`
  - `retry`
  - `other`

`thinking` 节点示例：

```json
{
  "step_id": "thinking-0",
  "type": "thinking",
  "thinking": "先分析用户意图。",
  "signature": "sig-1",
  "index": 0,
  "status": "running",
  "timestamp": "2026-03-23T12:00:00Z",
  "order": 1
}
```

`thinking` 节点结束示例：

```json
{
  "step_id": "thinking-0",
  "type": "thinking",
  "thinking": "先分析用户意图。",
  "signature": "sig-1",
  "index": 0,
  "status": "success",
  "timestamp": "2026-03-23T12:00:01Z",
  "order": 1
}
```

说明：

- 同一个 `thinking-*` 节点会先以 `status="running"` 持续增量更新
- 当思考结束时，服务端会对同一个 `step_id` 再发送一条 `trace_step(status="success")`
- 这条 `success` 更新就是“思考结束”的实时信号
- 如果用户在思考过程中主动停止，服务端不会补发 `thinking success`，而是直接进入 `trace_done.status="stopped"`

`trace_done`

- 仅在 `thinking_enabled=true` 时出现
- `status` 为 `completed` 或 `stopped`

`done`

- 整个 SSE 收尾
- 返回 `status` 和 `run_id`

## 会话详情

assistant 消息只返回：

```json
{
  "role": "assistant",
  "content": "我先查一下这个 IP。",
  "trace_steps": [
    {
      "step_id": "thinking-0",
      "type": "thinking",
      "thinking": "先分析用户意图。",
      "status": "success",
      "timestamp": "2026-03-23T12:00:00Z",
      "order": 1
    },
    {
      "step_id": "search-1",
      "type": "search",
      "status": "success",
      "message": "先搜索相关资料。",
      "timestamp": "2026-03-23T12:00:01Z",
      "order": 2
    }
  ],
  "sequence": 2,
  "created_at": "2026-03-23T12:00:03Z"
}
```

不再返回 `content_blocks`。
