# 前端单链展示约定

后端已经进一步简化：

- 非思考模式没有链，只会收到 `chunk`
- 思考模式下，所有非正文过程节点统一通过 `trace_step` 返回
- `thinking_block` 已废弃
- 历史回放不再依赖 `content_blocks`

## 前端处理原则

### 1. 实时流

思考模式事件顺序：

```text
conversation -> trace_step* -> thinking(success upsert)? -> chunk* -> trace_done -> done
```

前端只需要处理两类可视节点：

- `trace_step`
- `chunk`

### 2. trace_step.type

前端按 `trace_step.type` 渲染不同节点：

- `thinking`
- `tool_call`
- `tool_result`
- `search`
- `fetch`
- `retry`
- `other`

`thinking` 节点中的展示文本来自：

- `trace_step.thinking`

`thinking` 节点的结束信号：

- 同一个 `step_id` 会先收到若干次 `trace_step(status="running")`
- 思考结束时，后端会对同一个 `step_id` 再发一次 `trace_step(status="success")`
- 前端应更新原节点状态，不要新增第二个 thinking 节点

### 3. 历史回放

assistant 消息只保留：

- `content`
- `trace_steps`

历史链顺序直接按 `trace_steps.order` 恢复，不再读取 `content_blocks`。

### 4. 是否显示链

判断规则很简单：

- 当前轮收到任意一个 `trace_step`，就显示链
- 否则按普通消息渲染

### 5. 节点更新策略

- `trace_step` 必须按 `step_id` upsert
- 对 `thinking` 节点：
  - `running` 表示思考中
  - `success` 表示思考结束
- 如果整轮以 `trace_done.status="stopped"` 收尾，且某个 thinking 节点没有收到 `success`，前端应将其视为被中断而不是正常完成
