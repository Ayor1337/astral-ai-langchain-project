# AstralAI API 文档

## 概览

- 基础地址：`http://127.0.0.1:8000`
- Swagger：`/docs`
- ReDoc：`/redoc`

当前后端提供：

- `GET /`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
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
TITLE_AGENT_PROVIDER=anthropic
TITLE_AGENT_API_KEY=
TITLE_AGENT_BASE_URL=
TITLE_AGENT_MODEL=
SEARCH_PROVIDER=tavily
SEARCH_API_KEY=
SEARCH_BASE_URL=https://api.tavily.com
SEARCH_TIMEOUT_SECONDS=8
SEARCH_MAX_RESULTS=5
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/astral_ai
MEMORY_WINDOW_SIZE=8
MEMORY_SUMMARY_TRIGGER=12
JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_EXPIRE_SECONDS=604800
JWT_ALGORITHM=HS256
```

说明：

- `thinking_enabled=true` 当前仅支持 `anthropic`
- `thinking_enabled=false` 不会暴露任何链路事件
- `TITLE_AGENT_*` 为可选配置；仅用于首轮对话标题生成
- `SEARCH_*` 为联网搜索配置；只有请求 `search_enabled=true` 时才会用到
- 当 `search_enabled=true` 但未配置 `SEARCH_API_KEY` 时，请求会返回 `500`
- `JWT_SECRET_KEY` 用于本地账号 JWT 的签发与校验
- `JWT_EXPIRE_SECONDS` 默认为 `604800`
- `JWT_ALGORITHM` 当前固定为 `HS256`

## 通用约定

- 时间字段统一为 ISO 8601 字符串
- UUID 字段统一为字符串
- 除 `/`、`/docs`、`/redoc`、`/openapi.json`、`/api/auth/register`、`/api/auth/login` 外，其余接口都需要 `Authorization: Bearer <access_token>`
- 普通错误响应格式：

```json
{
  "detail": "conversation not found"
}
```

## 健康检查

### 请求

```http
GET /
```

### 响应

`200 OK`

```json
{
  "message": "AstralAI is running"
}
```

## 流式聊天

### 请求

```http
POST /api/chat/stream
Accept: text/event-stream
Content-Type: application/json
Authorization: Bearer <access_token>
```

### 请求体

```json
{
  "conversation_id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "message": "查一下 207.97.137.107",
  "thinking_enabled": true,
  "search_enabled": true
}
```

字段说明：

- `conversation_id`：可选，不传时服务端会隐式建会话
- `message`：必填
- `thinking_enabled`
  - `false`：只返回正文
  - `true`：正文之外的过程节点统一走 `trace_step`
- `search_enabled`
  - `false`：不向模型暴露联网搜索工具
  - `true`：允许模型在需要最新信息、事实核验、新闻或时间敏感问题时调用联网搜索

可能的 HTTP 状态码：

- `200`：成功建立 SSE
- `400`：当前 provider 不支持请求的能力
- `404`：`conversation_id` 不存在
- `500`：服务配置错误
- `502`：上游模型服务错误，或流启动前未产出任何事件

### SSE 时序

#### 非思考模式

```text
conversation -> chunk* -> conversation_title? -> done
```

#### 思考模式

```text
conversation -> trace_step* -> thinking(success upsert)? -> chunk* -> conversation_title? -> trace_done -> done
```

### SSE 事件

`conversation`

- 返回 `conversation_id`、`title`、`run_id`
- 这是首个事件，会立即返回；如果是新会话，这里的 `title` 通常还是默认值 `新对话`

`chunk`

- 返回正文文本分片
- 只承载 assistant 最终回答正文，不会携带搜索工具返回的原始 JSON

`conversation_title`

- 仅在“首轮对话”成功生成标题时出现
- 返回 `conversation_id`、`title`
- 事件会在 assistant 正文输出完成后、`done` 之前发送
- 后端会同步把该标题写回会话记录；后续列表和详情接口会返回更新后的标题
- 如果标题 agent 未配置、调用失败、assistant 没有正文，或用户中途停止，本事件不会出现，聊天主流程也不会报错

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

`search` 节点示例：

```json
{
  "step_id": "search-1",
  "type": "search",
  "query": "Astral AI 最新消息",
  "kind": "result_list",
  "status": "success",
  "result_count": 2,
  "timestamp": "2026-03-27T12:00:01Z",
  "order": 2,
  "payload": {
    "results": [
      {
        "title": "Astral AI",
        "url": "https://example.com/astral",
        "snippet": "Latest update"
      }
    ]
  }
}
```

说明：

- `search` 节点只在 `thinking_enabled=true` 且模型实际触发联网搜索时出现
- `status="running"` 表示正在搜索
- `status="success"` 表示搜索成功并带回结构化结果
- `status="error"` 表示搜索失败；聊天会降级继续进行，不会直接中断整轮 SSE
- 前端可将 `query` 渲染为搜索节点标题
- 前端可将 `payload.results` 渲染为该搜索节点下的子列表
- 子列表项建议只展示 `title`，并将其链接到 `url`
- `snippet` 在协议中保留，但不建议在思考链中直接展示

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
- 返回 `status`、`run_id` 和 `sources`
- `sources` 为本轮最终回答可展示的来源列表
- 当本轮未触发搜索、没有有效结果，或回答未引用来源时，`sources` 为空数组

`done` 示例：

```json
{
  "status": "completed",
  "run_id": "8bc85d87-ea36-46de-aeeb-d26c17e57ef3",
  "sources": [
    {
      "index": 1,
      "title": "Astral AI",
      "url": "https://example.com/astral",
      "snippet": "Latest update"
    }
  ]
}
```

来源约定：

- assistant 正文中的引用格式为数字引用，如 `[1][2]`
- `done.sources[].index` 与正文中的引用编号一一对应
- v1 只返回搜索摘要，不抓取网页正文

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

## 显式创建会话

推荐前端在点击“新建对话”时先创建空会话，再携带返回的 `conversation_id` 调用 `/api/chat/stream`。

### 请求

```http
POST /api/conversations
Authorization: Bearer <access_token>
```

### 响应

`201 Created`

```json
{
  "id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "title": "新对话",
  "summary": null,
  "created_at": "2026-03-17T00:00:00Z",
  "updated_at": "2026-03-17T00:00:00Z"
}
```

可能的 HTTP 状态码：

- `201`：创建成功
- `500`：服务配置错误

## 会话列表

### 请求

```http
GET /api/conversations
Authorization: Bearer <access_token>
```

### 响应

`200 OK`

```json
[
  {
    "id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
    "title": "测试会话",
    "summary": "摘要",
    "created_at": "2026-03-17T00:00:00Z",
    "updated_at": "2026-03-17T01:00:00Z"
  }
]
```

说明：

- 仅返回未软删除的会话
- 按 `updated_at` 倒序排列

## 获取会话详情

### 请求

```http
GET /api/conversations/{conversation_id}
Authorization: Bearer <access_token>
```

### 路径参数

- `conversation_id`：会话 ID

### 响应

`200 OK`

```json
{
  "id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "title": "测试会话",
  "summary": "摘要",
  "created_at": "2026-03-17T00:00:00Z",
  "updated_at": "2026-03-17T01:00:00Z",
  "messages": [
    {
      "role": "user",
      "content": "你好",
      "trace_steps": null,
      "sequence": 1,
      "created_at": "2026-03-17T00:00:00Z"
    },
    {
      "role": "assistant",
      "content": "你好，我在。",
      "trace_steps": null,
      "sequence": 2,
      "created_at": "2026-03-17T00:00:03Z"
    }
  ]
}
```

说明：

- 新建空会话时，`messages` 为空数组
- `role` 当前只会返回 `user` 和 `assistant`
- assistant 消息可能带 `trace_steps`

可能的 HTTP 状态码：

- `200`：查询成功
- `404`：会话不存在
- `500`：服务配置错误

## 更新会话标题

### 请求

```http
PATCH /api/conversations/{conversation_id}
Content-Type: application/json
Authorization: Bearer <access_token>
```

### 路径参数

- `conversation_id`：会话 ID

### 请求体

```json
{
  "title": "新标题"
}
```

### 响应

`200 OK`

```json
{
  "id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
  "title": "新标题",
  "summary": null,
  "created_at": "2026-03-17T00:00:00Z",
  "updated_at": "2026-03-17T01:00:00Z"
}
```

说明：

- 仅更新标题，不修改摘要和消息历史
- `title` 不能为空字符串

可能的 HTTP 状态码：

- `200`：更新成功
- `404`：会话不存在
- `422`：请求体校验失败
- `500`：服务配置错误

## 删除会话

### 请求

```http
DELETE /api/conversations/{conversation_id}
Authorization: Bearer <access_token>
```

### 路径参数

- `conversation_id`：会话 ID

### 响应

`204 No Content`

说明：

- 为软删除，不物理删除数据库记录
- 删除后不会再出现在会话列表中
- 后续访问详情或继续聊天会返回 `404`

可能的 HTTP 状态码：

- `204`：删除成功
- `404`：会话不存在
- `500`：服务配置错误

## 终止流式聊天

该接口用于终止当前正在进行的 `/api/chat/stream`。`run_id` 来自首个 `conversation` 事件。

### 请求

```http
POST /api/chat/runs/{run_id}/stop
Authorization: Bearer <access_token>
```

### 路径参数

- `run_id`：聊天运行 ID

### 响应

`202 Accepted`

```json
{
  "run_id": "8bc85d87-ea36-46de-aeeb-d26c17e57ef3",
  "status": "stop_requested"
}
```

可能的 HTTP 状态码：

- `202`：已接受终止请求
- `404`：聊天运行不存在或已结束

## 认证接口

### 注册

```http
POST /api/auth/register
Content-Type: application/json
```

```json
{
  "username": "alice_01",
  "nickname": "Alice",
  "password": "password123"
}
```

成功返回：

```json
{
  "access_token": "jwt-token",
  "token_type": "bearer",
  "expires_in": 604800,
  "user": {
    "id": "0f31cc7e-0ec7-4d8f-9baf-84f7072a2a98",
    "username": "alice_01",
    "nickname": "Alice",
    "created_at": "2026-03-28T12:00:00Z"
  }
}
```

### 登录

```http
POST /api/auth/login
Content-Type: application/json
```

```json
{
  "username": "alice_01",
  "password": "password123"
}
```

成功返回结构与注册相同。

### 当前用户

```http
GET /api/auth/me
Authorization: Bearer <access_token>
```
