# 前端 `thought_step` 需求说明书

## 背景与目标

当前后端能返回模型原始 `thinking`，但直接透传给前端时通常会表现为一整段不断变长的文本，缺少明显的步骤感。

本次改造的目标是：

- 前端实时收到离散的思考步骤，而不是原始大段文本。
- 思考步骤与工具轨迹分离：`thought_step` 表示前端可展示的思考步骤，`trace_step` 表示搜索、抓取、工具调用等执行轨迹。
- 刷新历史时，前端继续通过会话详情接口里的 `trace_steps` 回放整条链。

## 新事件契约

### 1. `thought_step`

仅在 `thinking_enabled=true` 且思考步骤整理成功时出现。

示例：

```json
{
  "step_id": "assistant-thought-0f31-2-1",
  "type": "thought",
  "status": "running",
  "title": "确定查询方向",
  "message": "先搜索可用的 IP 信息来源。",
  "timestamp": "2026-03-18T12:00:00+00:00",
  "order": -1000
}
```

字段说明：

- `step_id`：步骤唯一标识，前端必须按它做 upsert。
- `type`：固定为 `thought`。
- `status`：`running/success/skipped`。
- `title`：步骤标题，适合做粗粒度标签。
- `message`：步骤说明，适合直接显示。
- `timestamp`：事件时间。
- `order`：排序字段，值越小越靠前。

### 2. `trace_step`

继续承载执行轨迹，例如：

- `search`
- `fetch`
- `tool_call`
- `tool_result`
- `retry`

当 `thought_step` 生成失败或超时时，后端会回退为旧格式的单节点 `trace_step(type=thought)`。

## 状态机与渲染规则

### 1. `thought_step` 渲染规则

- 收到新的 `thought_step(step_id=X,status=running)` 时：
  - 若本地不存在该 `step_id`，新增一条思考步骤。
  - 若本地已存在该 `step_id`，按最新字段覆盖。
- 收到同一 `step_id` 的 `success` 时：
  - 将该步骤从“思考中”切换为“已完成”。
- 同一时间最多应有一条 `running` 的思考步骤。
- 首个 assistant `chunk` 不应先于首个 `thought_step` 渲染。

### 2. `trace_step` 渲染建议

推荐两种实现方式，前端可择一：

- 分区展示：
  - 上方显示思考步骤区，只消费 `thought_step`
  - 下方显示执行轨迹区，只消费 `trace_step`
- 统一时间线展示：
  - 将 `thought_step` 与 `trace_step` 都视为时间线节点
  - 但在视觉上区分“思考步骤”和“工具执行”

如果使用统一时间线，建议：

- `thought_step` 使用更轻量的视觉样式
- `trace_step` 保留搜索结果、抓取状态、工具名等结构化信息

## 历史回放规则

刷新会话历史时，不会重放 SSE，而是调用 `GET /api/conversations/{conversation_id}`。

历史思考链回放规则：

- 从 assistant 消息的 `trace_steps` 中筛选 `type=thought` 的节点
- 这些节点就是历史思考步骤
- 其他 `type` 继续按工具轨迹渲染

注意：

- 历史接口不新增 `thought_steps` 字段
- 前端不要依赖 assistant `content_blocks` 中存在原始 `thinking`

## 回退兼容规则

如果当前轮没有收到 `thought_step`，但收到了 `trace_step(type=thought)`，说明后端已回退到旧格式。

此时前端应：

- 继续显示链
- 将该 `trace_step(type=thought)` 当作单节点思考步骤处理
- 不因为缺少 `thought_step` 而隐藏整条链

兼容判断建议：

- 实时链是否显示：只要收到 `thought_step` 或 `trace_step`
- 思考区是否显示：
  - 优先使用 `thought_step`
  - 若没有 `thought_step`，再退回 `trace_step(type=thought)`

## 完整 SSE 示例

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

## 验收标准

- `thinking_enabled=true` 时，前端能正确渲染 `thought_step`。
- 首个 `chunk` 出现前，前端已经显示至少一条思考步骤。
- 当后一条步骤出现时，前一条步骤会从 `running` 切换为 `success`。
- 历史刷新后，前端能从 `trace_steps` 正确恢复 thought 节点。
- 后端回退到旧格式时，前端仍能显示单节点思考链。
- `trace_step` 与 `thought_step` 不会互相覆盖，界面不会出现重复或错位排序。
