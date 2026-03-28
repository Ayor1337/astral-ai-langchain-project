from __future__ import annotations

import asyncio
import unittest

from app.services.chat.stream_loop import (
    _StreamLoopState,
    _finalize_active_thinking_if_needed,
    _handle_text_block,
    _handle_trace_block,
    _wait_next_signal,
)


class StreamLoopHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_next_signal_returns_chunk_when_stream_yields_first(self):
        state = _StreamLoopState()

        async def stream():
            yield {"type": "text", "text": "first"}

        item_type, payload = await _wait_next_signal(
            state,
            stream=stream(),
            stop_event=asyncio.Event(),
            title_task=None,
        )

        self.assertEqual(item_type, "chunk")
        self.assertEqual(payload, {"type": "text", "text": "first"})

    async def test_wait_next_signal_returns_title_when_title_task_completes_first(self):
        state = _StreamLoopState()
        title_task = asyncio.create_task(asyncio.sleep(0, result="标题"))

        async def stream():
            await asyncio.sleep(0.1)
            yield {"type": "text", "text": "later"}

        item_type, payload = await _wait_next_signal(
            state,
            stream=stream(),
            stop_event=asyncio.Event(),
            title_task=title_task,
        )

        self.assertEqual(item_type, "title")
        self.assertIsNone(payload)

    async def test_wait_next_signal_returns_stop_when_stop_event_is_set(self):
        state = _StreamLoopState()
        stop_event = asyncio.Event()
        stop_event.set()

        async def stream():
            yield {"type": "text", "text": "never"}

        item_type, payload = await _wait_next_signal(
            state,
            stream=stream(),
            stop_event=stop_event,
            title_task=None,
        )

        self.assertEqual(item_type, "stop")
        self.assertIsNone(payload)
        self.assertTrue(state.stopped)

    async def test_wait_next_signal_returns_eof_when_stream_ends(self):
        state = _StreamLoopState()

        async def stream():
            if False:
                yield None

        item_type, payload = await _wait_next_signal(
            state,
            stream=stream(),
            stop_event=asyncio.Event(),
            title_task=None,
        )

        self.assertEqual(item_type, "eof")
        self.assertIsNone(payload)

    async def test_finalize_active_thinking_if_needed_returns_success_event_and_clears_active_step(self):
        state = _StreamLoopState(
            trace_state={
                "thinking-0": {
                    "step_id": "thinking-0",
                    "type": "thinking",
                    "status": "running",
                    "thinking": "先分析用户意图。",
                }
            },
            active_thinking_step_id="thinking-0",
        )

        event = _finalize_active_thinking_if_needed(state)

        self.assertEqual(event[0], "trace_step")
        self.assertEqual(event[1]["step_id"], "thinking-0")
        self.assertEqual(event[1]["type"], "thinking")
        self.assertEqual(event[1]["status"], "success")
        self.assertEqual(event[1]["thinking"], "先分析用户意图。")
        self.assertIn("timestamp", event[1])
        self.assertIsNone(state.active_thinking_step_id)

    async def test_finalize_active_thinking_if_needed_returns_none_for_non_running_step(self):
        state = _StreamLoopState(
            trace_state={
                "thinking-0": {
                    "step_id": "thinking-0",
                    "type": "thinking",
                    "status": "success",
                    "thinking": "已完成。",
                }
            },
            active_thinking_step_id="thinking-0",
        )

        event = _finalize_active_thinking_if_needed(state)

        self.assertIsNone(event)
        self.assertEqual(state.active_thinking_step_id, "thinking-0")

    async def test_handle_text_block_finalizes_thinking_before_emitting_chunk(self):
        state = _StreamLoopState(
            trace_state={
                "thinking-0": {
                    "step_id": "thinking-0",
                    "type": "thinking",
                    "status": "running",
                    "thinking": "先分析用户意图。",
                }
            },
            active_thinking_step_id="thinking-0",
        )

        events = list(
            _handle_text_block(
                {"type": "text", "text": "你好"},
                state,
                use_trace=True,
            )
        )

        self.assertEqual([name for name, _ in events], ["trace_step", "chunk"])
        self.assertEqual(events[0][1]["step_id"], "thinking-0")
        self.assertEqual(events[0][1]["status"], "success")
        self.assertEqual(events[1], ("chunk", {"content": "你好"}))
        self.assertEqual(state.assistant_text_chunks, ["你好"])

    async def test_handle_trace_block_emits_tool_result_then_tool_end(self):
        state = _StreamLoopState(trace_order=1)

        events = list(
            _handle_trace_block(
                {
                    "type": "tool_result",
                    "step_id": "call-1",
                    "tool_name": "add",
                    "output_json": '{"result":2}',
                },
                state,
            )
        )

        self.assertEqual([name for name, _ in events], ["trace_step", "trace_step"])
        self.assertEqual(events[0][1]["type"], "tool_result")
        self.assertEqual(events[1][1]["type"], "tool_end")
        self.assertEqual(events[1][1]["parent_step_id"], "call-1")
