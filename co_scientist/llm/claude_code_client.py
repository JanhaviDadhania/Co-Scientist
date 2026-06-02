"""Claude Code subprocess provider.

Routes every LLM call through `claude -p --output-format json` so co-scientist
uses the user's Claude Code subscription instead of an ANTHROPIC_API_KEY.

Pattern lifted from stanford-storm/storm/claudecode_lm.py. Differences from
that simpler text-only wrapper:

- We must honor `tools` + `tool_choice` because co-scientist's tool loop is
  the load-bearing scaffold (record_hypothesis, record_verdict, etc.).
- When `tool_choice = {"type": "tool", "name": X}`, we pass that tool's
  input_schema to `claude -p --json-schema <schema>`, which constrains the
  CLI to emit a JSON object matching the schema. We then synthesize an
  Anthropic-shaped `tool_use` block from the parsed JSON.
- When tools are present but tool_choice is auto/required, we ask the model
  to emit `{"tool_calls": [...]}` in its text and parse it out (best-effort).

Trade-offs vs the AnthropicClient:
- Each call shells out (~3-5s of `claude` cold-start). Use sparingly / with
  generous wall-clock budgets.
- No cache_control, no Anthropic batch API, no streaming.
- cost_usd is still estimated against the API price table — useful as a
  rough proxy for "what this would have cost on the API."
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from ..config import Config
from ..ids import transcript_id
from ..models import Transcript
from ..storage.artifacts import write_json
from ..storage.repos import sessions as sessions_repo
from ..storage.repos import transcripts as transcripts_repo
from .anthropic_client import (
    AgentCallSpec,
    AnthropicResponse,
    CallContext,
    _rough_token_count,
)
from .budgets import TokenBudget
from .openai_client import _Block, _Message, _Usage
from .retry import RetryPolicy
from .routing import estimate_cost_usd

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
DEFAULT_TIMEOUT_S = int(os.environ.get("CLAUDE_CODE_TIMEOUT_S", "400"))


class ClaudeCodeError(RuntimeError):
    pass


class ClaudeCodeClient:
    """Run LLM calls via the `claude` CLI in --print mode."""

    def __init__(
        self,
        cfg: Config,
        *,
        db: aiosqlite.Connection,
        budget: TokenBudget,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._budget = budget
        self._timeout_s = DEFAULT_TIMEOUT_S

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> AnthropicResponse:
        system_text, user_text, json_schema, forced_tool_name, tempfile_path = self._build_prompts(spec)

        est_in = est_input_tokens or _rough_token_count(spec)
        est_out = spec.max_output_tokens
        est_cost = estimate_cost_usd(
            model=spec.route.model, input_tokens=est_in, output_tokens=est_out
        )
        await self._budget.admit(
            ctx.agent, est_tokens=est_in + est_out, est_usd=est_cost
        )

        started = datetime.now(UTC)
        t0 = time.monotonic()

        try:
            raw = await self._invoke(
                system_text,
                user_text,
                model=spec.route.model,
                json_schema=json_schema,
                use_write_tool=tempfile_path is not None,
            )
        except BaseException:
            self._cleanup_tempfile(tempfile_path)
            await self._budget.settle(
                ctx.agent,
                est_tokens=est_in + est_out,
                est_usd=est_cost,
                actual_input_tokens=0,
                actual_output_tokens=0,
                actual_usd=0.0,
            )
            raise
        finished = datetime.now(UTC)

        file_data = self._read_and_cleanup_tempfile(tempfile_path)
        message = self._adapt(
            raw,
            spec.route.model,
            forced_tool_name=forced_tool_name,
            has_tools=bool(spec.tools),
            file_data=file_data,
        )
        in_tok = message.usage.input_tokens
        out_tok = message.usage.output_tokens
        cost_usd = estimate_cost_usd(
            model=spec.route.model, input_tokens=in_tok, output_tokens=out_tok
        )

        await self._budget.settle(
            ctx.agent,
            est_tokens=est_in + est_out,
            est_usd=est_cost,
            actual_input_tokens=in_tok,
            actual_output_tokens=out_tok,
            actual_usd=cost_usd,
        )

        trn_id = transcript_id()
        artifact = {
            "provider": "claude_code",
            "request": {
                "system": system_text,
                "prompt": user_text,
                "json_schema": json_schema,
                "model": spec.route.model,
                "forced_tool_name": forced_tool_name,
            },
            "response": {"raw": raw, "message": message.model_dump()},
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_ms": int((time.monotonic() - t0) * 1000),
        }
        artifact_path = await write_json(
            self._cfg, ctx.session_id, f"transcripts/{ctx.agent}", trn_id, artifact
        )

        t = Transcript(
            id=trn_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            agent=ctx.agent,
            action=ctx.action,
            model=spec.route.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read=0,
            cache_write=0,
            cost_usd=cost_usd,
            started_at=started,
            finished_at=finished,
            artifact_path=artifact_path,
        )
        await transcripts_repo.insert(self._db, t)
        await sessions_repo.add_usage(
            self._db, ctx.session_id, in_tok + out_tok, cost_usd
        )

        return AnthropicResponse(
            raw=message,
            transcript_id=trn_id,
            cost_usd=cost_usd,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read=0,
            cache_write=0,
        )

    # ------------------------- prompt construction ------------------------ #

    def _build_prompts(
        self, spec: AgentCallSpec
    ) -> tuple[str, str, dict | None, str | None, Path | None]:
        """Returns (system_text, user_text, json_schema_or_none, forced_tool_name_or_none, tempfile_path_or_none).

        When a forced tool is requested, we ask the model to use the Write tool
        to drop the tool-call JSON into a per-call tempfile. Write is a
        first-class Claude Code tool and is far more reliable than coercing
        the model into emitting raw JSON on stdout.
        """
        system_parts: list[str] = [b.text for b in spec.system_blocks if b.text]

        json_schema: dict | None = None
        forced_tool_name: str | None = None
        tempfile_path: Path | None = None

        if spec.tools:
            if spec.tool_choice and spec.tool_choice.get("type") == "tool":
                forced_tool_name = spec.tool_choice.get("name")

            if forced_tool_name:
                for tool in spec.tools:
                    if tool.get("name") == forced_tool_name:
                        schema = tool.get("input_schema") or {"type": "object"}
                        json_schema = schema  # passed to --json-schema as a hint
                        tempfile_path = Path(tempfile.gettempdir()) / (
                            f"co_scientist_call_{uuid.uuid4().hex}.json"
                        )
                        system_parts.append(
                            f"You have the Write tool available. Use it to write your "
                            f"response JSON to this exact file path:\n\n"
                            f"  {tempfile_path}\n\n"
                            f"The file MUST contain ONLY a single valid JSON object that "
                            f"conforms to the input schema of the tool '{forced_tool_name}':\n"
                            f"{json.dumps(schema, default=str)}\n\n"
                            f"Do NOT include markdown fences, code blocks, prose, or any "
                            f"explanatory text inside the file. The file's content must "
                            f"parse as raw JSON directly. After writing the file, you may "
                            f"write a short acknowledgement in your reply, but the file is "
                            f"what the system reads — your reply is ignored. Call Write "
                            f"exactly once with the full JSON object as the file_text."
                        )
                        break
            else:
                system_parts.append(self._format_tool_catalog(spec.tools))
                tc_type = (spec.tool_choice or {}).get("type", "auto")
                if tc_type in ("any", "required"):
                    system_parts.append(
                        "You MUST call exactly one of the tools listed above."
                    )
                system_parts.append(
                    'When you want to call a tool, respond with ONLY this JSON object '
                    '(no markdown fences, no extra text):\n'
                    '{"tool_calls": [{"name": "<tool_name>", "arguments": {<args matching that tool\'s input_schema>}}]}\n'
                    'If you do not want to call a tool, respond with plain text.'
                )

        system_text = "\n\n".join(p for p in system_parts if p).strip()

        user_parts: list[str] = []
        user_text = "\n\n".join(b.text for b in spec.user_blocks if b.text).strip()
        if user_text:
            user_parts.append(user_text)
        if spec.extra_messages:
            user_parts.append("\n--- Conversation so far ---")
            for m in spec.extra_messages:
                user_parts.append(self._format_message(m))
            user_parts.append("--- End of conversation ---\n")
            user_parts.append("Continue as the assistant. Respond now.")

        return system_text, "\n\n".join(user_parts).strip(), json_schema, forced_tool_name, tempfile_path

    def _format_tool_catalog(self, tools: list[dict[str, Any]]) -> str:
        lines = ["You have access to the following tools:"]
        for t in tools:
            lines.append(f"- name: {t.get('name', '')}")
            desc = t.get("description", "")
            if desc:
                lines.append(f"  description: {desc}")
            schema = t.get("input_schema", {})
            lines.append(f"  input_schema: {json.dumps(schema, default=str)}")
        return "\n".join(lines)

    def _format_message(self, m: dict[str, Any]) -> str:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            return f"[{role}] {content}"
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append(f"[{role}] {block}")
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(f"[{role}] {block.get('text', '')}")
                elif btype == "tool_use":
                    args = json.dumps(block.get("input", {}), default=str)
                    parts.append(
                        f"[{role}] (tool call) {block.get('name', '')}({args})"
                    )
                elif btype == "tool_result":
                    body = block.get("content", "")
                    if not isinstance(body, str):
                        body = json.dumps(body, default=str)
                    parts.append(
                        f"[tool_result for {block.get('tool_use_id', '')}] {body}"
                    )
                elif btype == "thinking":
                    continue
                else:
                    parts.append(f"[{role}:{btype}] {json.dumps(block, default=str)}")
            return "\n".join(parts)
        return f"[{role}] {content}"

    # ------------------------- subprocess invocation ---------------------- #

    async def _invoke(
        self,
        system: str,
        user: str,
        *,
        model: str,
        json_schema: dict | None,
        use_write_tool: bool = False,
    ) -> dict[str, Any]:
        # When we're routing a forced tool call through a tempfile, we have to
        # allow Claude Code's Write tool *and* bypass the permission prompt
        # (which would otherwise hang in non-interactive --print mode).
        tools_arg = "Write" if use_write_tool else ""
        args: list[str] = [
            CLAUDE_BIN,
            "-p",
            "--model", model,
            "--output-format", "json",
            "--no-session-persistence",
            "--tools", tools_arg,
        ]
        if use_write_tool:
            args += ["--permission-mode", "bypassPermissions"]
        if system:
            args += ["--system-prompt", system]
        if json_schema is not None:
            args += ["--json-schema", json.dumps(json_schema, default=str)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(user.encode()),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError as e:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise ClaudeCodeError(
                f"claude -p timed out after {self._timeout_s}s"
            ) from e

        if proc.returncode != 0:
            raise ClaudeCodeError(
                f"claude -p exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
            )

        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCodeError(
                f"claude -p returned non-JSON: {stdout[:300]!r}"
            ) from e
        if raw.get("is_error"):
            raise ClaudeCodeError(
                f"claude -p reported error: "
                f"{raw.get('api_error_status') or raw.get('result')!r}"
            )
        return raw

    # ------------------------- response adaptation ------------------------ #

    def _adapt(
        self,
        raw: dict[str, Any],
        model: str,
        *,
        forced_tool_name: str | None,
        has_tools: bool,
        file_data: Any = None,
    ) -> _Message:
        text = (raw.get("result") or "").strip()
        usage = raw.get("usage", {}) or {}

        blocks: list[_Block] = []
        tool_used = False

        if forced_tool_name:
            # Prefer the JSON the model wrote via the Write tool to a tempfile.
            # That path is much more reliable than coercing it to emit raw JSON
            # on stdout. We only fall back to stdout parsing if the file is
            # missing or invalid.
            args_obj = file_data if isinstance(file_data, (dict, list)) else None
            if args_obj is None:
                args_obj = self._parse_json_object(text)
            if args_obj is not None:
                if not isinstance(args_obj, dict):
                    args_obj = {"_args": args_obj}
                blocks.append(_Block(
                    type="tool_use",
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=forced_tool_name,
                    input=args_obj,
                ))
                tool_used = True
            else:
                blocks.append(_Block(type="text", text=text))
        elif has_tools:
            tcs = self._extract_tool_calls(text)
            if tcs:
                for tc in tcs:
                    args = tc.get("arguments") or {}
                    if not isinstance(args, dict):
                        args = {"_args": args}
                    blocks.append(_Block(
                        type="tool_use",
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=tc.get("name", ""),
                        input=args,
                    ))
                tool_used = True
            else:
                blocks.append(_Block(type="text", text=text))
        else:
            blocks.append(_Block(type="text", text=text))

        stop_reason = "tool_use" if tool_used else "end_turn"
        return _Message(
            content=blocks,
            stop_reason=stop_reason,
            usage=_Usage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
            model=model,
            id=str(raw.get("session_id", "")),
        )

    @staticmethod
    def _read_and_cleanup_tempfile(path: Path | None) -> Any:
        """Read JSON from the tempfile the model was instructed to write to.

        Returns the parsed JSON value, or None if the file is missing /
        empty / invalid. Always removes the file on the way out.
        """
        if path is None:
            return None
        try:
            data = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        data = data.strip()
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # Sometimes the model wraps the JSON in a ```json fence even though
            # we asked it not to. Strip and retry.
            fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", data)
            if fence:
                try:
                    return json.loads(fence.group(1))
                except json.JSONDecodeError:
                    pass
            return None

    @staticmethod
    def _cleanup_tempfile(path: Path | None) -> None:
        """Best-effort tempfile removal used on error paths."""
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _parse_json_object(text: str) -> Any | None:
        """Try parsing `text` as JSON; tolerates a leading/trailing fence or prose.

        Used when --json-schema forced the model to emit a single JSON object.
        """
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Strip ```json fences if present.
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass
        # Last resort: first balanced {...} object.
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        start = -1
        return None

    @staticmethod
    def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
        """Find a `{"tool_calls": [...]}` block in `text` and return the list."""
        if "tool_calls" not in text:
            return []
        # Scan for balanced JSON objects and check which one has tool_calls.
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start : i + 1]
                    if "tool_calls" in candidate:
                        try:
                            parsed = json.loads(candidate)
                        except json.JSONDecodeError:
                            start = -1
                            continue
                        tcs = parsed.get("tool_calls")
                        if isinstance(tcs, list):
                            return tcs
                    start = -1
        return []
