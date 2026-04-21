from __future__ import annotations

import base64
import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from imagination_runtime.chat_web_research import fetch_web_context
from imagination_runtime.paths import resolve_root_path


JsonDict = Dict[str, Any]
ModelGenerateFn = Callable[[List[Dict[str, str]], int], str]


_SESSIONS_LOCK = Lock()
_SESSIONS: Dict[str, "AgentSessionState"] = {}


_DEFAULT_SYSTEM_PROMPT = """You are Imagination Composer, an autonomous coding agent.
Return ONLY valid JSON for each turn with this schema:
{
  "thought": "short reasoning",
  "tool_call": {
    "name": "read_file|write_file|run_shell|web_search|capture_ui",
    "args": { ... }
  }
}

When done, return:
{
  "thought": "short reasoning",
  "final_answer": "what was done and next action",
  "summary_report": {
    "files_modified": [{"path": "...", "why": "..."}],
    "commands_run": ["..."],
    "captures": [{"artifact_id": "...", "kind": "screenshot|video"}]
  }
}
"""


@dataclass
class DiffProposal:
    proposal_id: str
    rel_path: str
    old_content: str
    new_content: str
    reason: str = ""


@dataclass
class CaptureArtifact:
    artifact_id: str
    kind: str
    abs_path: str
    mime_type: str
    rel_path: str


@dataclass
class AgentSessionState:
    session_id: str
    workspace_root: str
    proposals: Dict[str, DiffProposal] = field(default_factory=dict)
    artifacts: Dict[str, CaptureArtifact] = field(default_factory=dict)
    modified_files: Dict[str, str] = field(default_factory=dict)
    commands_run: List[str] = field(default_factory=list)


def _normalize_session_id(raw: Optional[str]) -> str:
    base = (raw or "").strip()
    if base:
        return re.sub(r"[^a-zA-Z0-9._-]", "_", base)[:80]
    return f"session_{uuid.uuid4().hex[:10]}"


def _default_workspace_for_session(session_id: str) -> str:
    root = resolve_root_path(None)
    base = Path(root) / "temp" / "agent_workspaces" / session_id
    base.mkdir(parents=True, exist_ok=True)
    return str(base.resolve())


def get_or_create_session(session_id: Optional[str], workspace_root: Optional[str]) -> AgentSessionState:
    sid = _normalize_session_id(session_id)
    with _SESSIONS_LOCK:
        state = _SESSIONS.get(sid)
        if state is not None:
            return state

        if workspace_root:
            ws = str(Path(workspace_root).expanduser().resolve())
            Path(ws).mkdir(parents=True, exist_ok=True)
        else:
            ws = _default_workspace_for_session(sid)

        state = AgentSessionState(session_id=sid, workspace_root=ws)
        _SESSIONS[sid] = state
        return state


def workspace_tree(workspace_root: str, *, max_depth: int = 5, max_entries: int = 1000) -> JsonDict:
    root = Path(workspace_root).resolve()
    entries = 0

    def build(node: Path, depth: int) -> Optional[JsonDict]:
        nonlocal entries
        if entries >= max_entries:
            return None
        if depth > max_depth:
            return {"name": node.name, "path": str(node.relative_to(root)) if node != root else "", "type": "dir", "truncated": True, "children": []}

        entries += 1
        rel = "" if node == root else str(node.relative_to(root)).replace("\\", "/")
        if node.is_file():
            return {"name": node.name, "path": rel, "type": "file", "size": node.stat().st_size}

        children: List[JsonDict] = []
        try:
            for child in sorted(node.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if child.name.startswith(".git"):
                    continue
                built = build(child, depth + 1)
                if built is not None:
                    children.append(built)
                if entries >= max_entries:
                    break
        except PermissionError:
            pass

        return {"name": node.name or ".", "path": rel, "type": "dir", "children": children}

    tree = build(root, 0)
    return {"root": str(root), "tree": tree, "truncated": entries >= max_entries}


def list_workspace_tree(session_id: Optional[str], workspace_root: Optional[str]) -> JsonDict:
    state = get_or_create_session(session_id, workspace_root)
    payload = workspace_tree(state.workspace_root)
    payload["session_id"] = state.session_id
    return payload


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text_safe(path: Path, *, max_bytes: int = 300_000) -> Tuple[str, bool]:
    if not path.exists():
        return "", False
    raw = path.read_bytes()
    clipped = len(raw) > max_bytes
    if clipped:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace"), clipped


def _extract_json_object(text: str) -> Optional[JsonDict]:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
        if m:
            t = m.group(1).strip()
    try:
        loaded = json.loads(t)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            loaded = json.loads(t[start : end + 1])
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return None
    return None


def _safe_rel(workspace: Path, input_path: str) -> Tuple[Path, str]:
    rel = (input_path or "").strip().replace("\\", "/").lstrip("/")
    if not rel:
        raise ValueError("path is required")
    candidate = (workspace / rel).resolve()
    try:
        candidate.relative_to(workspace)
    except Exception as exc:
        raise ValueError("path escapes workspace") from exc
    rel_norm = str(candidate.relative_to(workspace)).replace("\\", "/")
    return candidate, rel_norm


def _diff_unified(old_text: str, new_text: str, rel_path: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        )
    )


def apply_proposals(session_id: str, proposal_ids: List[str]) -> JsonDict:
    state = get_or_create_session(session_id, None)
    applied: List[JsonDict] = []
    skipped: List[str] = []
    ws = Path(state.workspace_root).resolve()
    for pid in proposal_ids:
        prop = state.proposals.get(pid)
        if prop is None:
            skipped.append(pid)
            continue
        abs_path, rel_path = _safe_rel(ws, prop.rel_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(prop.new_content, encoding="utf-8")
        state.modified_files[rel_path] = prop.reason or "Applied diff proposal"
        applied.append({"proposal_id": pid, "path": rel_path})
        del state.proposals[pid]
    return {"session_id": state.session_id, "applied": applied, "skipped": skipped}


def capture_artifact(session_id: str, artifact_id: str) -> Optional[CaptureArtifact]:
    state = get_or_create_session(session_id, None)
    return state.artifacts.get(artifact_id)


class AgenticLoop:
    def __init__(
        self,
        *,
        model_generate: ModelGenerateFn,
        session_id: Optional[str] = None,
        workspace_root: Optional[str] = None,
        max_tool_iters: int = 28,
        max_new_tokens: int = 1024,
        confirm_apply: bool = False,
        allow_network_tools: bool = True,
    ) -> None:
        self.state = get_or_create_session(session_id, workspace_root)
        self.model_generate = model_generate
        self.max_tool_iters = max(1, min(64, int(max_tool_iters)))
        self.max_new_tokens = max(128, min(4096, int(max_new_tokens)))
        self.confirm_apply = bool(confirm_apply)
        self.allow_network_tools = bool(allow_network_tools)
        self.workspace = Path(self.state.workspace_root).resolve()

    def run(self, *, user_prompt: str, messages: Optional[List[Dict[str, str]]] = None) -> Iterator[JsonDict]:
        convo: List[Dict[str, str]] = [{"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}]
        if messages:
            for m in messages:
                role = (m.get("role") or "").strip().lower()
                if role in ("user", "assistant"):
                    convo.append({"role": role, "content": str(m.get("content") or "")})
        elif user_prompt.strip():
            convo.append({"role": "user", "content": user_prompt.strip()})
        else:
            convo.append({"role": "user", "content": "Inspect workspace and propose first coding step."})

        yield {"type": "session", "session_id": self.state.session_id, "workspace_root": self.state.workspace_root}
        yield {"type": "workspace_tree", **workspace_tree(self.state.workspace_root)}

        final_answer = ""
        for _ in range(self.max_tool_iters):
            raw = self.model_generate(convo, self.max_new_tokens)
            parsed = _extract_json_object(raw)
            if not parsed:
                err = "Model did not return valid JSON object"
                yield {"type": "error", "message": err}
                convo.append({"role": "assistant", "content": raw})
                convo.append(
                    {
                        "role": "system",
                        "content": "Return valid JSON exactly following the schema. No markdown fences.",
                    }
                )
                continue

            thought = str(parsed.get("thought") or "").strip()
            if thought:
                yield {"type": "thought", "text": thought}

            if parsed.get("final_answer"):
                final_answer = str(parsed.get("final_answer") or "").strip()
                break

            tool_call = parsed.get("tool_call")
            if not isinstance(tool_call, dict):
                yield {"type": "error", "message": "Missing tool_call or final_answer"}
                convo.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                convo.append(
                    {
                        "role": "system",
                        "content": "Provide either tool_call or final_answer in the next JSON response.",
                    }
                )
                continue

            tool_name = str(tool_call.get("name") or "").strip()
            args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
            call_id = uuid.uuid4().hex[:10]
            yield {"type": "tool_call", "id": call_id, "name": tool_name, "args": args}

            ok, result = self._execute_tool(tool_name, args)
            yield {"type": "tool_result", "id": call_id, "name": tool_name, "ok": ok, "data": result}

            convo.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            convo.append(
                {
                    "role": "system",
                    "content": f"Tool result for {tool_name}: {json.dumps(result, ensure_ascii=False)[:4000]}",
                }
            )

            if tool_name == "run_shell" and not ok:
                convo.append(
                    {
                        "role": "system",
                        "content": (
                            "The shell command failed. Diagnose stderr/stdout and propose a corrected command "
                            "or an alternate fix path."
                        ),
                    }
                )

        if not final_answer:
            final_answer = "Reached tool iteration budget; reporting current status and pending actions."

        report = self._build_summary_report()
        yield {"type": "summary", "report": report}
        yield {"type": "final", "text": final_answer}

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, JsonDict]:
        try:
            if tool_name == "read_file":
                return True, self._tool_read_file(args)
            if tool_name == "write_file":
                return True, self._tool_write_file(args)
            if tool_name == "run_shell":
                return self._tool_run_shell(args)
            if tool_name == "web_search":
                return True, self._tool_web_search(args)
            if tool_name == "capture_ui":
                return self._tool_capture_ui(args)
            return False, {"error": f"unknown_tool: {tool_name}"}
        except Exception as exc:
            return False, {"error": str(exc)}

    def _tool_read_file(self, args: Dict[str, Any]) -> JsonDict:
        path = str(args.get("path") or "")
        abs_path, rel = _safe_rel(self.workspace, path)
        text, clipped = _read_text_safe(abs_path)
        return {
            "path": rel,
            "exists": abs_path.exists(),
            "content": text,
            "truncated": clipped,
            "sha256": _sha256_text(text),
        }

    def _tool_write_file(self, args: Dict[str, Any]) -> JsonDict:
        path = str(args.get("path") or "")
        content = str(args.get("content") or "")
        reason = str(args.get("reason") or "").strip()
        abs_path, rel = _safe_rel(self.workspace, path)
        old_text, _ = _read_text_safe(abs_path, max_bytes=2_000_000)
        diff_text = _diff_unified(old_text, content, rel)
        proposal_id = uuid.uuid4().hex[:12]
        event: JsonDict = {
            "proposal_id": proposal_id,
            "path": rel,
            "diff": diff_text,
            "old_sha256": _sha256_text(old_text),
            "new_sha256": _sha256_text(content),
            "applied": False,
        }
        if self.confirm_apply:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            self.state.modified_files[rel] = reason or "Agent write_file"
            event["applied"] = True
        else:
            self.state.proposals[proposal_id] = DiffProposal(
                proposal_id=proposal_id,
                rel_path=rel,
                old_content=old_text,
                new_content=content,
                reason=reason,
            )
        return {"type": "diff_preview", **event}

    def _tool_run_shell(self, args: Dict[str, Any]) -> Tuple[bool, JsonDict]:
        allow_shell = (os.getenv("IMAGINATION_AGENT_ALLOW_SHELL") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if not allow_shell:
            return False, {"error": "run_shell disabled by IMAGINATION_AGENT_ALLOW_SHELL=0"}

        command = str(args.get("command") or args.get("cmd") or "").strip()
        if not command:
            return False, {"error": "command is required"}
        cwd_raw = str(args.get("cwd") or "").strip()
        timeout_ms = int(args.get("timeout_ms") or 90_000)
        timeout_sec = max(1, min(600, timeout_ms // 1000))

        if cwd_raw:
            cwd_abs, _ = _safe_rel(self.workspace, cwd_raw)
            cwd = str(cwd_abs)
        else:
            cwd = str(self.workspace)

        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except Exception:
            argv = [command]

        self.state.commands_run.append(command)
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=False,
        )
        stdout = (completed.stdout or "")[:10000]
        stderr = (completed.stderr or "")[:10000]
        ok = completed.returncode == 0
        return ok, {
            "command": command,
            "cwd": cwd,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "status": "success" if ok else "fail",
        }

    def _tool_web_search(self, args: Dict[str, Any]) -> JsonDict:
        if not self.allow_network_tools:
            return {"error": "network tools disabled for this request"}
        query = str(args.get("query") or "").strip()
        max_results = int(args.get("max_results") or 8)
        snippets = fetch_web_context(query, max_results=max(1, min(12, max_results)))
        return {"query": query, "results": snippets}

    def _tool_capture_ui(self, args: Dict[str, Any]) -> Tuple[bool, JsonDict]:
        capture_on = (os.getenv("IMAGINATION_AGENT_ENABLE_CAPTURE") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if not capture_on:
            return False, {"error": "capture_ui disabled by IMAGINATION_AGENT_ENABLE_CAPTURE=0"}

        if not self.allow_network_tools:
            return False, {"error": "network tools disabled for this request"}

        url = str(args.get("url") or "").strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            return False, {"error": "url must start with http:// or https://"}

        allow_hosts = (os.getenv("IMAGINATION_AGENT_CAPTURE_ALLOW_HOSTS") or "").strip()
        if allow_hosts:
            from urllib.parse import urlparse

            host = (urlparse(url).hostname or "").lower()
            allowed = {h.strip().lower() for h in allow_hosts.split(",") if h.strip()}
            if host not in allowed:
                return False, {"error": f"host not allowlisted: {host}"}

        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as exc:
            return False, {"error": f"playwright import failed: {exc}"}

        out_dir = Path(self.state.workspace_root) / ".captures"
        out_dir.mkdir(parents=True, exist_ok=True)
        shot_id = uuid.uuid4().hex[:12]
        image_path = out_dir / f"{shot_id}.png"
        video_path: Optional[Path] = None

        want_video = bool(args.get("video"))
        wait_ms = int(args.get("wait_ms") or 1200)
        wait_ms = max(0, min(15_000, wait_ms))
        width = int(args.get("width") or 1400)
        height = int(args.get("height") or 900)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context_kwargs: Dict[str, Any] = {"viewport": {"width": width, "height": height}}
                if want_video:
                    context_kwargs["record_video_dir"] = str(out_dir)
                context = browser.new_context(**context_kwargs)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                if wait_ms:
                    page.wait_for_timeout(wait_ms)
                page.screenshot(path=str(image_path), full_page=True)
                if want_video:
                    page.wait_for_timeout(1000)
                    video = page.video
                else:
                    video = None
                page.close()
                context.close()
                browser.close()
                if video is not None:
                    try:
                        video_path = Path(video.path())
                    except Exception:
                        video_path = None
        except Exception as exc:
            return False, {"error": f"capture failed: {exc}"}

        rel_img = str(image_path.relative_to(self.workspace)).replace("\\", "/")
        img_artifact = CaptureArtifact(
            artifact_id=shot_id,
            kind="screenshot",
            abs_path=str(image_path),
            mime_type="image/png",
            rel_path=rel_img,
        )
        self.state.artifacts[shot_id] = img_artifact

        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload: JsonDict = {
            "kind": "screenshot",
            "artifact_id": shot_id,
            "mime_type": "image/png",
            "relative_path": rel_img,
            "base64": image_b64,
        }
        if video_path and video_path.exists():
            vid_id = uuid.uuid4().hex[:12]
            rel_vid = str(video_path.relative_to(self.workspace)).replace("\\", "/") if self.workspace in video_path.parents else video_path.name
            self.state.artifacts[vid_id] = CaptureArtifact(
                artifact_id=vid_id,
                kind="video",
                abs_path=str(video_path),
                mime_type="video/webm",
                rel_path=rel_vid,
            )
            payload["video"] = {
                "artifact_id": vid_id,
                "mime_type": "video/webm",
                "relative_path": rel_vid,
            }
        return True, {"type": "media", **payload}

    def _build_summary_report(self) -> JsonDict:
        files_modified = [{"path": path, "why": why} for path, why in sorted(self.state.modified_files.items())]
        captures = [
            {"artifact_id": c.artifact_id, "kind": c.kind, "path": c.rel_path}
            for c in self.state.artifacts.values()
        ]
        pending = [
            {"proposal_id": p.proposal_id, "path": p.rel_path, "reason": p.reason}
            for p in self.state.proposals.values()
        ]
        return {
            "session_id": self.state.session_id,
            "workspace_root": self.state.workspace_root,
            "files_modified": files_modified,
            "commands_run": list(self.state.commands_run),
            "captures": captures,
            "pending_proposals": pending,
        }
