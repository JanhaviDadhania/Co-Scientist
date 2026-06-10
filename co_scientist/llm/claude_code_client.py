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
from .schema_validate import validate_payload

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

    # Total attempts for a forced tool call: 1 initial + 2 validation retries.
    MAX_FORCED_TOOL_ATTEMPTS = 3

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> AnthropicResponse:
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

        total_in = 0
        total_out = 0
        attempts_log: list[dict[str, Any]] = []
        message: _Message | None = None
        forced_tool_name: str | None = None
        json_schema: dict | None = None
        validation_errors: list[str] = []

        try:
            for attempt in range(self.MAX_FORCED_TOOL_ATTEMPTS):
                # New prompts (and a fresh tempfile path) per attempt.
                system_text, user_text, json_schema, forced_tool_name, tempfile_path = (
                    self._build_prompts(spec)
                )
                if validation_errors:
                    user_text += (
                        "\n\n=== PREVIOUS ATTEMPT FAILED VALIDATION ===\n"
                        f"Your previous reply did not produce a valid "
                        f"`{forced_tool_name}` payload:\n"
                        + "\n".join(f"- {e}" for e in validation_errors)
                        + "\nUse the Write tool to write the CORRECTED raw-JSON "
                        "payload to the file path given in the system prompt. "
                        "Fix every error listed above."
                    )

                try:
                    raw = await self._invoke(
                        system_text,
                        user_text,
                        model=spec.route.model,
                        # The file carries the payload; never constrain stdout.
                        json_schema=None,
                        use_write_tool=tempfile_path is not None,
                    )
                except BaseException:
                    self._cleanup_tempfile(tempfile_path)
                    raise

                file_data = self._read_and_cleanup_tempfile(tempfile_path)
                message = self._adapt(
                    raw,
                    spec.route.model,
                    forced_tool_name=forced_tool_name,
                    has_tools=bool(spec.tools),
                    file_data=file_data,
                    forced_tool_schema=json_schema,
                    tools=spec.tools,
                )
                total_in += message.usage.input_tokens
                total_out += message.usage.output_tokens
                attempts_log.append({
                    "attempt": attempt + 1,
                    "request": {
                        "system": system_text,
                        "prompt": user_text,
                        "json_schema": json_schema,
                        "model": spec.route.model,
                        "forced_tool_name": forced_tool_name,
                    },
                    "response": {"raw": raw, "message": message.model_dump()},
                })

                if forced_tool_name is None:
                    break

                payload = None
                for b in message.content:
                    if (
                        getattr(b, "type", None) == "tool_use"
                        and getattr(b, "name", "") == forced_tool_name
                    ):
                        payload = getattr(b, "input", None)
                if payload is None:
                    validation_errors = [
                        f"no `{forced_tool_name}` payload was found — the JSON "
                        f"file was missing, empty, or unparseable"
                    ]
                    continue
                validation_errors = validate_payload(json_schema, payload)
                if not validation_errors:
                    break

            if forced_tool_name is not None and validation_errors:
                raise ClaudeCodeError(
                    f"forced tool {forced_tool_name!r} failed validation after "
                    f"{len(attempts_log)} attempt(s): {'; '.join(validation_errors[:8])}"
                )
        except BaseException:
            # Settle with whatever was actually consumed, persist the attempt
            # transcripts (save everything), then propagate loudly.
            cost = estimate_cost_usd(
                model=spec.route.model, input_tokens=total_in, output_tokens=total_out
            )
            await self._budget.settle(
                ctx.agent,
                est_tokens=est_in + est_out,
                est_usd=est_cost,
                actual_input_tokens=total_in,
                actual_output_tokens=total_out,
                actual_usd=cost,
            )
            if attempts_log:
                try:
                    await self._record_transcript(
                        ctx, spec, attempts_log, started, t0,
                        in_tok=total_in, out_tok=total_out, cost_usd=cost,
                    )
                except Exception:  # noqa: BLE001 — never mask the original error
                    pass
            raise

        assert message is not None
        finished = datetime.now(UTC)
        _ = finished

        in_tok = total_in
        out_tok = total_out
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

        trn_id, artifact_path = await self._record_transcript(
            ctx, spec, attempts_log, started, t0,
            in_tok=in_tok, out_tok=out_tok, cost_usd=cost_usd,
        )

        _ = artifact_path
        return AnthropicResponse(
            raw=message,
            transcript_id=trn_id,
            cost_usd=cost_usd,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read=0,
            cache_write=0,
        )

    async def _record_transcript(
        self,
        ctx: CallContext,
        spec: AgentCallSpec,
        attempts_log: list[dict[str, Any]],
        started: datetime,
        t0: float,
        *,
        in_tok: int,
        out_tok: int,
        cost_usd: float,
    ) -> tuple[str, str]:
        """Write the full multi-attempt transcript artifact + DB row."""
        finished = datetime.now(UTC)
        trn_id = transcript_id()
        artifact = {
            "provider": "claude_code",
            "n_attempts": len(attempts_log),
            "attempts": attempts_log,
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
        return trn_id, artifact_path

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
                        json_schema = schema
                        # FILE HANDSHAKE: the model commits by Writing the JSON
                        # payload to a per-call tempfile. Write is Claude Code's
                        # most-practiced native tool — far more reliable than
                        # coercing structured output onto stdout. The wrapper
                        # reads the file back, validates it against the schema,
                        # and retries with the exact errors on failure.
                        tempfile_path = (
                            Path(tempfile.gettempdir())
                            / f"co_scientist_{uuid.uuid4().hex}.json"
                        )
                        system_parts.append(
                            f"==================================================\n"
                            f"HARD CONSTRAINT — COMMIT VIA FILE WRITE\n"
                            f"==================================================\n\n"
                            f"You must call the tool '{forced_tool_name}'. In this "
                            f"environment you do that by WRITING A FILE: use your "
                            f"Write tool to create the file\n\n"
                            f"    {tempfile_path}\n\n"
                            f"containing EXACTLY ONE JSON object — the arguments "
                            f"payload for '{forced_tool_name}'. The file must be raw "
                            f"JSON: no markdown fences, no commentary, nothing else.\n\n"
                            f"Automated code reads that file and VALIDATES it against "
                            f"the JSON schema below. If the file is missing or the "
                            f"payload is invalid you will be re-invoked with the "
                            f"validation errors; after repeated failures the pipeline "
                            f"halts.\n\n"
                            f"After writing the file, end your reply with the single "
                            f"line:\n"
                            f"COMMITTED {forced_tool_name}\n\n"
                            f"JSON schema for the payload (types and required fields "
                            f"are enforced):\n"
                            f"{json.dumps(schema, default=str)}"
                        )
                        break
            else:
                system_parts.append(self._format_tool_catalog(spec.tools))
                tc_type = (spec.tool_choice or {}).get("type", "auto")
                if tc_type in ("any", "required"):
                    system_parts.append(
                        "You MUST call exactly one of the tools listed above."
                    )
                # Two output formats — JSON tool_calls on stdout for cheap
                # search/lookup tools, and the FILE HANDSHAKE for "record_*"
                # commit tools (large structured payloads): the model Writes
                # the payload JSON to a per-call tempfile, which is far more
                # reliable than emitting a multi-paragraph JSON object or a
                # fragile markdown template on stdout.
                record_tools = [
                    t for t in spec.tools
                    if (t.get("name") or "").startswith("record_")
                ]
                system_parts.append(
                    'When you want to call a SEARCH or LOOKUP tool, respond with ONLY this JSON object '
                    '(no markdown fences, no extra text):\n'
                    '{"tool_calls": [{"name": "<tool_name>", "arguments": {<args matching that tool\'s input_schema>}}]}\n'
                    'If you do not want to call any tool, respond with plain text.'
                )
                if record_tools:
                    tempfile_path = (
                        Path(tempfile.gettempdir())
                        / f"co_scientist_{uuid.uuid4().hex}.json"
                    )
                    names = ", ".join(f"`{t['name']}`" for t in record_tools)
                    schemas_block = "\n\n".join(
                        f"### {t['name']}\n"
                        f"{json.dumps(t.get('input_schema') or {}, default=str)}"
                        for t in record_tools
                    )
                    system_parts.append(
                        f"------------------------------------------------------\n"
                        f"SPECIAL CASE — committing via {names} (structured-commit)\n"
                        f"------------------------------------------------------\n\n"
                        f"These commit tools carry large multi-field payloads. Do "
                        f"NOT use the `{{\"tool_calls\": [...]}}` JSON format for "
                        f"them and do NOT print the payload to stdout. Instead, "
                        f"when you are ready to commit, use your Write tool to "
                        f"create the file\n\n"
                        f"    {tempfile_path}\n\n"
                        f"containing exactly one JSON object of the form:\n"
                        f'{{"tool": "<commit tool name>", "payload": {{ ...arguments '
                        f"matching that tool's input_schema... }}}}\n\n"
                        f"The file must be raw JSON — no markdown fences, no "
                        f"commentary. After writing it, end your reply with the "
                        f"single line:\n"
                        f"COMMITTED <commit tool name>\n\n"
                        f"If you want to call a search tool instead this turn, use "
                        f"the JSON form above and do not write the file.\n\n"
                        f"Payload schemas:\n{schemas_block}"
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
        forced_tool_schema: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> _Message:
        text = (raw.get("result") or "").strip()
        usage = raw.get("usage", {}) or {}

        blocks: list[_Block] = []
        tool_used = False

        if forced_tool_name:
            # Primary: the FILE HANDSHAKE — the model Wrote the payload JSON to
            # a per-call tempfile and `file_data` carries it. Fallbacks (markdown
            # sections, JSON-in-text) cover models that answered on stdout
            # anyway; the validation+retry loop in `call()` is the gate that
            # decides whether whatever we recovered is actually acceptable.
            args_obj: Any | None = None
            if isinstance(file_data, dict):
                # Tolerate the {"tool": ..., "payload": ...} wrapper used by
                # the auto-path instructions even in forced mode.
                if isinstance(file_data.get("payload"), dict) and "tool" in file_data:
                    args_obj = file_data["payload"]
                else:
                    args_obj = file_data
            elif isinstance(file_data, list):
                args_obj = file_data
            if args_obj is None:
                args_obj = self._parse_markdown_payload(text, forced_tool_schema)
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
            # FILE HANDSHAKE first: a record_* commit Written to the tempfile.
            if isinstance(file_data, dict):
                name: str | None = None
                payload: dict | None = None
                if "tool" in file_data and isinstance(file_data.get("payload"), dict):
                    name = str(file_data["tool"])
                    payload = file_data["payload"]
                else:
                    # Bare payload: attribute it to the agent's single record_* tool.
                    record_names = [
                        t.get("name") or "" for t in (tools or [])
                        if (t.get("name") or "").startswith("record_")
                    ]
                    if len(record_names) == 1:
                        name, payload = record_names[0], file_data
                if name and payload is not None:
                    blocks.append(_Block(
                        type="tool_use",
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=name,
                        input=payload,
                    ))
                    tool_used = True
            if tool_used:
                pass
            elif (tcs := self._extract_tool_calls(text)):
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
                # No tool_calls JSON found — try the markdown-commit path for any
                # `record_*` tools in the catalog. This lets the model commit a
                # large structured payload via markdown sections instead of JSON.
                record_match = None
                for t in (tools or []):
                    name = t.get("name") or ""
                    if not name.startswith("record_"):
                        continue
                    schema = t.get("input_schema") or {}
                    parsed = self._parse_markdown_payload(text, schema)
                    if parsed:
                        record_match = (name, parsed)
                        break
                if record_match is not None:
                    name, parsed = record_match
                    blocks.append(_Block(
                        type="tool_use",
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=name,
                        input=parsed,
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

    # ------------------------------------------------------------------ #
    # Markdown payload builder + parser (primary path for forced tools)  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_markdown_template(tool_name: str, schema: dict) -> str:
        """Emit a markdown skeleton the model should mirror.

        Each top-level property in `schema.properties` becomes a `## <name>`
        section. Strings get a one-line placeholder; arrays-of-strings get two
        bullet placeholders; arrays-of-objects get two `### <name> <n>` blocks
        with `- key: value` lines.
        """
        props = (schema or {}).get("properties") or {}
        required = set((schema or {}).get("required") or [])
        out: list[str] = []
        for name, spec in props.items():
            marker = " (required)" if name in required else ""
            t = (spec or {}).get("type", "string")
            desc = (spec or {}).get("description", "")
            out.append(f"## {name}{marker}")
            if t == "array":
                items_spec = (spec or {}).get("items") or {}
                if items_spec.get("type") == "object":
                    inner_props = list((items_spec.get("properties") or {}).keys())
                    for n in (1, 2):
                        out.append(f"### {name} {n}")
                        for k in inner_props:
                            out.append(f"- {k}: <value>")
                else:
                    out.append("- <item 1>")
                    out.append("- <item 2>")
            else:
                hint = f"<{desc}>" if desc else f"<{t} value>"
                out.append(hint)
            out.append("")
        return "\n".join(out).rstrip()

    @staticmethod
    def _normalize_section_key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    @classmethod
    def _parse_markdown_payload(cls, text: str, schema: dict | None) -> dict | None:
        """Parse a structured-markdown response into a dict matching `schema`.

        Tolerant to header capitalization, trailing colons, surrounding prose,
        and missing sections. Returns None if no recognizable sections were
        found; otherwise returns a best-effort dict with whatever was parsed.
        """
        if not text or not isinstance(schema, dict):
            return None
        props = schema.get("properties") or {}
        if not props:
            return None
        # Constrain to the marker-bounded region if present.
        if "BEGIN ANSWER" in text:
            text = text.split("BEGIN ANSWER", 1)[1]
        if "END ANSWER" in text:
            text = text.split("END ANSWER", 1)[0]
        name_for = {cls._normalize_section_key(k): k for k in props.keys()}
        h2_pat = re.compile(r"(?m)^[ \t]*##[ \t]+(.+?)[ \t]*:?[ \t]*$\n")
        matches = list(h2_pat.finditer(text))
        if not matches:
            return None
        result: dict[str, Any] = {}
        for i, m in enumerate(matches):
            header = m.group(1).strip()
            # Strip any "(required)" / "(optional)" annotation if the model echoed it.
            header = re.sub(r"\s*\((?:required|optional)\)\s*$", "", header, flags=re.IGNORECASE)
            canon = name_for.get(cls._normalize_section_key(header))
            if canon is None:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            # Strip placeholder-style content the model may have left in.
            if body.startswith("<") and body.endswith(">") and "\n" not in body:
                continue
            result[canon] = cls._parse_section_body(body, props[canon])
        return result or None

    @classmethod
    def _parse_section_body(cls, body: str, spec: dict) -> Any:
        t = (spec or {}).get("type", "string")
        if t == "string":
            return body
        if t == "integer":
            mm = re.search(r"-?\d+", body)
            return int(mm.group()) if mm else 0
        if t == "number":
            mm = re.search(r"-?\d+(?:\.\d+)?", body)
            return float(mm.group()) if mm else 0.0
        if t == "boolean":
            return body.strip().lower() in ("true", "yes", "1")
        if t == "array":
            items_spec = (spec or {}).get("items") or {"type": "string"}
            if items_spec.get("type") == "object":
                return cls._parse_array_of_objects(body, items_spec)
            return cls._parse_bullets(body)
        return body

    @staticmethod
    def _parse_bullets(body: str) -> list[str]:
        out: list[str] = []
        for line in body.splitlines():
            mm = re.match(r"\s*(?:[-*+•]|\d+[.)])\s+(.+)", line)
            if mm:
                v = mm.group(1).strip()
                if v and not (v.startswith("<") and v.endswith(">")):
                    out.append(v)
        return out

    @classmethod
    def _parse_array_of_objects(cls, body: str, items_spec: dict) -> list[dict]:
        objs: list[dict] = []
        h3_pat = re.compile(r"(?m)^[ \t]*###[ \t]+(.+?)[ \t]*$\n")
        matches = list(h3_pat.finditer(body))
        if matches:
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
                obj = cls._parse_kv_pairs(body[start:end])
                if obj:
                    objs.append(obj)
            return objs
        obj = cls._parse_kv_pairs(body)
        return [obj] if obj else []

    @staticmethod
    def _parse_kv_pairs(body: str) -> dict:
        """Parse '- key: value' or 'key: value' lines into a dict (lowercased keys)."""
        out: dict[str, Any] = {}
        for line in body.splitlines():
            mm = re.match(r"\s*(?:[-*+]\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)", line)
            if mm:
                key = mm.group(1).strip().lower()
                val = mm.group(2).strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                if val.startswith("<") and val.endswith(">"):
                    continue
                out[key] = val
        return out

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
