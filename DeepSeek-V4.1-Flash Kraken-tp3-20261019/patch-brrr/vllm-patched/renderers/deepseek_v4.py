# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.tokenizers.deepseek_v41_encoding import (
    merge_tool_messages,
    sort_tool_results_by_call_order,
)
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams


def _tool_call_id(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function")
    function_id = function.get("id", "") if isinstance(function, dict) else ""
    return call.get("id") or function_id or ""


def _order_messages_like_v41_prompt(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    if not all(isinstance(message, dict) for message in messages):
        return messages
    markers: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        marker: dict[str, Any] = {"role": role, "content": index}
        if role == "tool":
            marker["tool_call_id"] = message.get("tool_call_id", "")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            marker["tool_calls"] = [{"id": _tool_call_id(call)} for call in tool_calls]
        if message.get("task") is not None:
            marker["task"] = message["task"]
        markers.append(marker)

    order: list[int] = []
    for merged in sort_tool_results_by_call_order(merge_tool_messages(markers)):
        blocks = merged.get("content_blocks")
        if blocks is None:
            order.append(merged["content"])
            continue
        for block in blocks:
            order.append(
                block["content"] if block["type"] == "tool_result" else block["text"]
            )

    if sorted(order) != list(range(len(messages))):
        return messages
    return [messages[index] for index in order]


class DeepseekV4Renderer(BaseRenderer[DeepseekV4Tokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: DeepseekV4Tokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def _order_for_prompt(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        if self.model_config.tokenizer_mode == "deepseek_v41":
            return _order_messages_like_v41_prompt(messages)
        return messages

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        messages = self._order_for_prompt(messages)
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        messages = self._order_for_prompt(messages)
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
