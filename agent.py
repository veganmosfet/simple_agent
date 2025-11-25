#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from html.parser import HTMLParser
import atexit
import ssl 

# -----------------------------
# Simple terminal color helpers
# -----------------------------
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def colorize(text: str, color: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Colors.RESET}"


def colorize_prompt(text: str, color: str, using_libedit: bool, enabled: bool = True) -> str:
    """Color a prompt safely for readline.

    - With libedit (macOS default), avoid coloring to prevent display glitches.
    - With GNU readline, wrap non-printing ANSI sequences with \001/\002 so
      readline can correctly calculate cursor positions and render history.
    """
    if not enabled:
        return text
    if using_libedit:
        return text

    # Mark non-printing sequences so GNU readline won't count them
    start_np = "\001"
    end_np = "\002"
    return f"{start_np}{color}{end_np}{text}{start_np}{Colors.RESET}{end_np}"


def reasoning_to_text(reasoning_content: Any) -> str:
    """Normalize reasoning_content (thinking traces) to a printable string."""

    def _to_text(val: Any) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            parts: List[str] = []
            for item in val:
                t = _to_text(item)
                if t:
                    parts.append(t)
            return "\n".join(parts) if parts else None
        if isinstance(val, dict):
            if val.get("type") == "text" and "text" in val:
                return _to_text(val.get("text"))
            if "content" in val:
                return _to_text(val.get("content"))
            if "text" in val:
                return _to_text(val.get("text"))
            if "message" in val:
                return _to_text(val.get("message"))
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
        try:
            return str(val)
        except Exception:
            return None

    text = _to_text(reasoning_content)
    if not text:
        return ""
    return str(text).strip()


# -----------------------------
# Line editing (readline) setup
# -----------------------------
def _write_history_safe(readline_mod, path: str) -> None:
    try:
        readline_mod.write_history_file(path)
    except Exception:
        pass


def init_line_editing() -> bool:
    """Initialize readline and return True if using libedit.

    On macOS, libedit's prompt handling can miscount ANSI escape sequences,
    causing history navigation to visually concatenate lines instead of
    replacing them. We detect libedit so we can avoid coloring the input
    prompt later.
    """
    is_libedit = False
    try:
        try:
            import gnureadline as readline  # Prefer GNU readline if installed
        except Exception:
            import readline  # libedit or gnureadline
        doc = getattr(readline, "__doc__", "") or ""
        is_libedit = "libedit" in doc
        # Prefer emacs-style editing so arrows/backspace behave consistently
        if is_libedit:
            # macOS often uses libedit; use compatible bindings
            readline.parse_and_bind("bind -e")  # emacs mode
            readline.parse_and_bind("set editing-mode emacs")
            readline.parse_and_bind("set keymap emacs-standard")
            readline.parse_and_bind("tab: complete")
        else:
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind("set editing-mode emacs")

        # Basic history across runs
        histfile = os.path.expanduser("~/.agent_history")
        try:
            readline.read_history_file(histfile)
        except Exception:
            pass
        atexit.register(_write_history_safe, readline, histfile)
    except Exception:
        # If readline is unavailable, continue without line editing
        is_libedit = False
    return is_libedit


# -----------------------------
# HTML to plain text converter
# -----------------------------
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._in_style = False
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        if tag == "style":
            self._in_style = True
        if tag == "script":
            self._in_script = True
        if tag in {"br", "hr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "style":
            self._in_style = False
        if tag == "script":
            self._in_script = False
        if tag in {"p", "div", "section", "article", "header", "footer", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data):
        # Skip CSS and script contents
        if self._in_style or self._in_script:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        # Join and normalize excessive newlines
        raw = " ".join(self.parts)
        # Collapse repeated spaces, preserve newlines
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join([line for line in lines if line])


def html_to_text(html: str) -> str:
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# -----------------------------
# Native tools
# -----------------------------
def tool_bash(command: str, timeout: int = 30, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Run a bash command and return stdout/stderr/exit_code."""
    try:
        # Use bash -lc to support pipes and shell features
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "exit_code": -1,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Tool error: {e}",
            "exit_code": -1,
        }


def tool_webfetch(
    url: str,
    user_agent: str = "Mozilla/5.0 (Agent)",
    timeout: int = 30,
    allow_self_signed: bool = False,
) -> Dict[str, Any]:
    """Fetch URL and return plain text extracted from HTML.
       Optionally allow self-signed certificates (UNSAFE, for testing only)."""
    try:
        req = Request(url, headers={"User-Agent": user_agent})
        # Create SSL context
        context = None
        if url.lower().startswith("https") and allow_self_signed:
            context = ssl._create_unverified_context()
        with urlopen(req, timeout=timeout, context=context) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            data = resp.read().decode(charset, errors="replace")
            text = html_to_text(data)
            return {
                "status": resp.status,
                "text": text,
            }
    except HTTPError as e:
        return {"status": e.code, "text": f"HTTP error: {e}"}
    except URLError as e:
        return {"status": -1, "text": f"Network error: {e}"}
    except Exception as e:
        return {"status": -1, "text": f"Tool error: {e}"}
    

def tool_readfile(filename: str, max_bytes: int, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Read the first max_bytes of a text file located in base_dir (current directory)."""
    if not filename:
        return {"error": "filename is required"}
    if max_bytes <= 0:
        return {"error": "max_bytes must be positive"}

    cwd = os.path.abspath(base_dir or os.getcwd())
    target = os.path.abspath(os.path.join(cwd, filename))

    if os.path.dirname(target) != cwd:
        return {"error": "readfile can only access files in the current directory"}
    if not os.path.exists(target):
        return {"error": f"File not found: {filename}"}
    if not os.path.isfile(target):
        return {"error": f"Not a regular file: {filename}"}

    try:
        with open(target, "rb") as fh:
            data = fh.read(max_bytes)
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    return {"filename": filename, "bytes_read": len(data), "content": text}


# -----------------------------
# MCP over HTTP (JSON-RPC 2.0)
# -----------------------------
class MCPHttpClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: int = 30, debug_tools: bool = False, colors_on: bool = True):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.debug_tools = debug_tools
        self.colors_on = colors_on
        self._id = 0

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        data_bytes = json.dumps(payload).encode("utf-8")
        if self.debug_tools:
            print(colorize(f"[debug] MCP -> {method}", Colors.YELLOW, self.colors_on))
            print(colorize(json.dumps(payload, indent=2), Colors.DIM, self.colors_on))
        try:
            req = Request(self.base_url, headers=self._headers(), data=data_bytes, method="POST")
            with urlopen(req, timeout=self.timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                charset = resp.headers.get_content_charset() or "utf-8"
                raw = resp.read().decode(charset, errors="replace")
                data: Dict[str, Any]
                if "text/event-stream" in content_type or raw.startswith("event:"):
                    # Parse SSE: collect 'data:' lines and join
                    data_lines: List[str] = []
                    for line in raw.splitlines():
                        if line.startswith("data: "):
                            data_lines.append(line[len("data: "):])
                    sse_payload = "\n".join(data_lines).strip()
                    try:
                        data = json.loads(sse_payload) if sse_payload else {"raw": raw}
                    except Exception:
                        data = {"raw": raw}
                else:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        data = {"raw": raw}
                if self.debug_tools:
                    print(colorize("[debug] MCP <- response", Colors.YELLOW, self.colors_on))
                    print(colorize(json.dumps(data, indent=2), Colors.DIM, self.colors_on))
                return data
        except HTTPError as e:
            body = None
            try:
                body_bytes = e.read()
                body = body_bytes.decode("utf-8", errors="replace") if body_bytes else None
            except Exception:
                body = None
            return {"error": {"code": e.code, "message": str(e), "body": body}}
        except URLError as e:
            return {"error": {"code": -1, "message": str(e)}}
        except Exception as e:
            return {"error": {"code": -1, "message": str(e)}}


def tool_mcp_list_tools_http(client: MCPHttpClient) -> Dict[str, Any]:
    return client.call("tools/list")


def tool_mcp_call_tool_http(client: MCPHttpClient, name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return client.call("tools/call", {"name": name, "arguments": arguments or {}})


# -----------------------------
# OpenAI-compatible client wrapper
# -----------------------------
class OpenAICompat:
    def __init__(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        debug_llm: bool = False,
        provider: str = "openai",
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
    ):
        self.debug_llm = debug_llm
        self.mode = None  # "v1" (new client)
        self.client = None
        self._base_url = base_url
        # Prefer explicit api_key, then provider-specific envs, then OPENAI_API_KEY
        self._api_key = (
            api_key
            or (os.getenv("AZURE_OPENAI_API_KEY") if provider == "azure" else None)
            or os.getenv("OPENAI_API_KEY")
            or ''
        )
        self._provider = provider
        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Lazily initialize client to avoid import errors before first use
        if self.client is None:
            try:
                if self._provider == "azure":
                    # Prefer AzureOpenAI client to set correct headers and endpoint handling
                    from openai import AzureOpenAI  # type: ignore
                    if not self._azure_endpoint:
                        raise RuntimeError("Azure endpoint is required. Provide --azure-endpoint or AZURE_OPENAI_ENDPOINT.")
                    def _normalize_azure_endpoint(u: str) -> str:
                        u = (u or "").strip()
                        if not u:
                            return u
                        # Remove trailing '/openai' if present; SDK adds its own paths
                        v = u.rstrip('/')
                        if v.endswith('/openai'):
                            v = v[:-len('/openai')]
                        return v
                    self.client = AzureOpenAI(
                        api_key=self._api_key,
                        api_version=self._azure_api_version,
                        azure_endpoint=_normalize_azure_endpoint(self._azure_endpoint),
                    )
                else:
                    from openai import OpenAI  # type: ignore
                    self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)
                self.mode = "v1"
            except Exception as e:
                raise RuntimeError(f"OpenAI SDK not available or misconfigured: {e}. Please install 'openai' package and check provider settings.")

        if self.mode == "v1":
            payload = {
                "model": model,
                "messages": messages,
            }
            # Temperature handling: some Azure deployments only allow default=1
            if self._provider == "azure":
                if temperature == 1:
                    payload["temperature"] = 1
                # else: omit temperature to use server default
            else:
                payload["temperature"] = temperature
            if self._provider == "azure":
                # Azure (2024/2025 APIs) expect max_completion_tokens
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice
            if reasoning_effort and not (self._provider == "azure"):
                extra_body = payload.setdefault("extra_body", {})
                extra_body["reasoning"] = {"effort": reasoning_effort}
            if self.debug_llm:
                print(colorize("[debug] request payload:", Colors.DIM))
                print(colorize(json.dumps(payload, indent=2), Colors.DIM))
            try:
                resp = self.client.chat.completions.create(**payload)
            except Exception as e:
                msg = str(e)
                if self._provider == "azure" and ("Resource not found" in msg or "404" in msg):
                    raise RuntimeError(
                        "Azure OpenAI: Resource not found. Verify endpoint (omit trailing /openai), API version, and that --model matches your deployment name."
                    ) from e
                raise
            # Convert to a dict for uniform downstream handling
            return json.loads(resp.model_dump_json())
        else:
            raise RuntimeError("OpenAI client not initialized")


# -----------------------------
# Agent implementation
# -----------------------------
#SYSTEM_PROMPT = """
# You are a helpful terminal agent.
#  - Use tools when appropriate.
#  - When calling a tool, respond ONLY with a JSON dictionary that matches the function schema.
#  - Prefer safe, read-only commands unless explicitly asked.
#  - Keep outputs concise and relevant.
#""".strip()

SYSTEM_PROMPT = """
# You are a helpful terminal agent.
  - Use tools when appropriate.
  - Prefer safe, read-only commands unless explicitly asked.
  - Keep outputs concise and relevant.
""".strip()

def build_tools(dynamic_mcp_tools: Optional[List[Dict[str, Any]]] = None, include_bash: bool = True, readfile_bytes: int = 4096) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if include_bash:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a bash command. Returns stdout/stderr/exit_code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"],
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "webfetch",
                "description": "Fetch a URL and return plain text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"],
                },
            },
        }
    )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "readfile",
                "description": f"Read the first {readfile_bytes} bytes of a text file in the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"}
                    },
                    "required": ["filename"],
                },
            },
        }
    )

    if dynamic_mcp_tools:
        tools.extend(dynamic_mcp_tools)

    return tools


def execute_tool_call(call: Dict[str, Any], opts: argparse.Namespace, colors_on: bool, mcp_client: Optional[MCPHttpClient], mcp_tools_map: Dict[str, Dict[str, Any]]) -> str:
    name = call.get("function", {}).get("name")
    raw_args = call.get("function", {}).get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except Exception:
        args = {"_raw": raw_args}

    # Resolve potential name aliases for MCP tools (underscore/hyphen/case variants)
    dispatch_name = name
    alias_note = ""
    if name not in ("bash", "webfetch", "readfile") and mcp_tools_map:
        if name not in mcp_tools_map:
            def _candidates(n: str) -> List[str]:
                base = [n]
                if "_" in n:
                    base.append(n.replace("_", "-"))
                if "-" in n:
                    base.append(n.replace("-", "_"))
                # lower-case variants
                nl = n.lower()
                if nl != n:
                    base.append(nl)
                if "_" in nl:
                    base.append(nl.replace("_", "-"))
                if "-" in nl:
                    base.append(nl.replace("-", "_"))
                # de-duplicate preserving order
                seen = set()
                out = []
                for x in base:
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
                return out

            for cand in _candidates(name or ""):
                if cand in mcp_tools_map:
                    dispatch_name = cand
                    if name and cand and cand != name:
                        alias_note = f" -> {cand}"
                    break

    # Always print a concise, single-line announcement of the tool call
    def _truncate_one_line(text: str, max_len: int = 160) -> str:
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)
        text = text.replace("\n", " ").replace("\r", " ")
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    summary = ""
    if dispatch_name == "bash":
        summary = _truncate_one_line(args.get("command", ""))
    elif dispatch_name == "webfetch":
        summary = _truncate_one_line(args.get("url", ""))
    else:
        # For MCP or unknown tools, show compact JSON of arguments
        try:
            summary = _truncate_one_line(args)
        except Exception:
            summary = ""
    shown_name = (name or "") + alias_note
    shown_name = shown_name.strip()
    print(colorize(f"[tool] {shown_name} {summary}".strip(), Colors.CYAN, colors_on))

    # Debug: show tool input
    if getattr(opts, "debug_tools", False):
        print(colorize(f"[debug] tool call -> {name}()", Colors.MAGENTA, colors_on))
        print(colorize(json.dumps(args, indent=2), Colors.DIM, colors_on))

    result: Dict[str, Any]
    if dispatch_name == "bash":
        if not getattr(opts, "bash_enabled", False):
            result = {"error": "'bash' tool disabled"}
        else:
            result = tool_bash(
                command=args.get("command", ""),
                timeout=int(args.get("timeout", 30)),
                cwd=args.get("cwd"),
            )
    elif dispatch_name == "webfetch":
        result = tool_webfetch(
            url=args.get("url", ""),
            user_agent=args.get("user_agent", "Mozilla/5.0 (Agent)"),
            timeout=int(args.get("timeout", 30)),
            allow_self_signed=bool(getattr(opts, "insecure", False)),
        )
    elif dispatch_name == "readfile":
        result = tool_readfile(
            filename=args.get("filename", ""),
            max_bytes=int(getattr(opts, "readfile_bytes", 4096)),
            base_dir=os.getcwd(),
        )
    elif dispatch_name in mcp_tools_map:
        if mcp_client is None:
            result = {"error": "MCP client not initialized"}
        else:
            # Pass arguments through exactly as provided by the model.
            result = tool_mcp_call_tool_http(mcp_client, dispatch_name, args)
    else:
        result = {"error": f"Unknown tool: {name}"}

    # Debug: show tool output
    if getattr(opts, "debug_tools", False):
        print(colorize("[debug] tool output:", Colors.MAGENTA, colors_on))
        print(colorize(json.dumps(result, indent=2), Colors.DIM, colors_on))

    # Return compact JSON string for the model
    return json.dumps(result)


def agent_loop(opts: argparse.Namespace) -> None:
    colors_on = not opts.no_color
    show_reasoning = not getattr(opts, "no_reasoning", False)
    error_prefix = colorize("assistant > [error] ", Colors.RED, colors_on)

    # Initialize OpenAI-compatible client
    # Backward compatibility: --debug enables both llm and tools logs
    if getattr(opts, "debug", False):
        setattr(opts, "debug_llm", True)
        setattr(opts, "debug_tools", True)

    client = OpenAICompat(
        base_url=opts.api_base,
        api_key=opts.api_key,
        debug_llm=getattr(opts, "debug_llm", False),
        provider=getattr(opts, "model_provider", "openai"),
        azure_endpoint=getattr(opts, "azure_endpoint", None),
        azure_api_version=getattr(opts, "azure_api_version", None),
    )

    # Optional MCP initialization: fetch tool list and convert to model tools
    mcp_client: Optional[MCPHttpClient] = None
    mcp_tools_map: Dict[str, Dict[str, Any]] = {}
    dynamic_mcp_tools: List[Dict[str, Any]] = []
    if opts.mcp_url:
        mcp_client = MCPHttpClient(opts.mcp_url, opts.mcp_token, debug_tools=getattr(opts, "debug_tools", False), colors_on=colors_on)
        mcp_resp = tool_mcp_list_tools_http(mcp_client)
        # Expect JSON-RPC with 'result', but handle direct 'tools' for flexibility
        tools_obj = None
        if isinstance(mcp_resp, dict):
            if "result" in mcp_resp:
                res = mcp_resp["result"]
                if isinstance(res, dict):
                    tools_obj = res.get("tools") or res.get("items")
                elif isinstance(res, list):
                    tools_obj = res
            elif "tools" in mcp_resp:
                tools_obj = mcp_resp.get("tools")
            elif "error" in mcp_resp and getattr(opts, "debug_tools", False):
                err = mcp_resp.get("error")
                print(colorize(f"[debug] MCP tools/list error: {err}", Colors.RED, colors_on))
        if isinstance(tools_obj, list):
            for t in tools_obj:
                if not isinstance(t, dict):
                    continue
                name = t.get("name")
                if not name:
                    continue
                desc = t.get("description") or "MCP tool"
                params = t.get("inputSchema") or t.get("input_schema") or t.get("parameters")
                # Fallback schema
                if not isinstance(params, dict):
                    params = {"type": "object", "properties": {}, "additionalProperties": True}
                dynamic_mcp_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params,
                    },
                })
                mcp_tools_map[name] = {"description": desc, "schema": params}

    tools = build_tools(
        dynamic_mcp_tools,
        include_bash=getattr(opts, "bash_enabled", False),
        readfile_bytes=int(getattr(opts, "readfile_bytes", 4096)),
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": opts.system_prompt or SYSTEM_PROMPT}]

    # Startup summary
    # Enable CLI line editing (arrow keys, history) where possible
    using_libedit = init_line_editing()

    print(colorize("Simple LLM Agent (type 'exit' to quit)", Colors.BOLD, colors_on))
    print(colorize(f"Model: {opts.model}", Colors.GREEN, colors_on))
    native_tools = (["bash"] if getattr(opts, "bash_enabled", False) else []) + ["readfile", "webfetch"]
    mcp_tool_names = sorted(list(mcp_tools_map.keys()))
    if mcp_tool_names:
        print(colorize(f"MCP tools: {', '.join(mcp_tool_names)}", Colors.YELLOW, colors_on))
    else:
        print(colorize("MCP tools: none", Colors.YELLOW, colors_on))
    print(colorize(f"Native tools: {', '.join(native_tools)}", Colors.CYAN, colors_on))
    print(colorize("System prompt:", Colors.MAGENTA, colors_on))
    print(colorize(messages[0]["content"], Colors.DIM, colors_on))

    def emit_reasoning(reasoning_value: Any) -> None:
        if not show_reasoning:
            return
        text = reasoning_to_text(reasoning_value)
        if not text:
            return
        prefix = "thinking > "
        lines = text.splitlines() or [""]
        print(colorize(prefix + lines[0], Colors.DIM, colors_on))
        for line in lines[1:]:
            print(colorize(" " * len(prefix) + line, Colors.DIM, colors_on))

    def pick_choice(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the first choice or None after emitting a user-visible error."""
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            err = resp.get("error") or resp.get("message") or "No choices returned by model."
            print(error_prefix + str(err))
            return None
        if len(choices) > 1 and getattr(opts, "debug_llm", False):
            print(colorize(f"[debug] multiple choices returned ({len(choices)}); using first", Colors.YELLOW, colors_on))
        choice = choices[0] or {}
        if not isinstance(choice, dict):
            print(error_prefix + "Invalid choice payload from model.")
            return None
        return choice

    while True:
        try:
            # Use uncolored prompt with libedit to avoid display glitches
            prompt_using_libedit = using_libedit
            prompt = colorize_prompt("you > ", Colors.CYAN, prompt_using_libedit, colors_on)
            user_input = input(prompt)
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input or user_input.strip().lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        # First call
        response = client.chat(
            model=opts.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            reasoning_effort=getattr(opts, "reasoning_effort", None),
        )
        if getattr(opts, "debug_llm", False):
            print(colorize("[debug] raw response:", Colors.DIM, colors_on))
            print(colorize(json.dumps(response, indent=2), Colors.DIM, colors_on))

        # Handle tool calls loop
        while True:
            choice = pick_choice(response)
            if choice is None:
                break
            message = choice.get("message") or {}
            if not isinstance(message, dict):
                print(error_prefix + "Invalid message payload from model.")
                break
            tool_calls = message.get("tool_calls") or []
            emit_reasoning(message.get("reasoning_content"))

            if tool_calls:
                # Show any assistant text that came with the tool calls
                if message.get("content"):
                    print(colorize("assistant > ", Colors.GREEN, colors_on) + str(message.get("content")))
                # Echo back the assistant message containing tool_calls to satisfy
                # providers (e.g., Azure) that require the tool response to follow
                # a preceding assistant message with tool_calls.
                assistant_with_tools: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_with_tools)
                for tc in tool_calls:
                    tool_output = execute_tool_call(tc, opts, colors_on, mcp_client, mcp_tools_map)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "content": tool_output,
                        }
                    )

                # Ask the model to continue after tool results
                response = client.chat(
                    model=opts.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=opts.temperature,
                    max_tokens=opts.max_tokens,
                    reasoning_effort=getattr(opts, "reasoning_effort", None),
                )
                if getattr(opts, "debug_llm", False):
                    print(colorize("[debug] raw response:", Colors.DIM, colors_on))
                    print(colorize(json.dumps(response, indent=2), Colors.DIM, colors_on))
                continue
            else:
                # Final assistant message
                content = message.get("content", "")
                print(colorize("assistant > ", Colors.GREEN, colors_on) + content)
                assistant_final: Dict[str, Any] = {"role": "assistant", "content": content}
                messages.append(assistant_final)
                break


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple LLM agent with tools and optional MCP stubs.")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Model name or Azure deployment name")
    p.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL (non-Azure)")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"), help="API key (OpenAI or Azure)")
    p.add_argument("--model-provider", choices=["openai", "azure"], default=os.getenv("MODEL_PROVIDER", "openai"), help="Model provider: openai or azure")
    p.add_argument("--azure-endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure OpenAI endpoint, e.g. https://<resource>.openai.azure.com")
    p.add_argument("--azure-api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"), help="Azure OpenAI API version")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max tokens in responses")
    p.add_argument("--readfile-bytes", type=int, default=4096, help="Number of bytes readfile tool returns from a file")
    p.add_argument("--system-prompt", default=None, help="Custom system prompt")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=os.getenv("REASONING_EFFORT", "medium"), help="Reasoning effort hint sent to the model (default: medium)")
    p.add_argument("--debug", action="store_true", help="Enable both LLM and tool debug logs (convenience)")
    p.add_argument("--debug-llm", action="store_true", help="Debug raw LLM traffic (requests/responses)")
    p.add_argument("--debug-tools", action="store_true", help="Debug tool calls and I/O (incl. MCP JSON-RPC)")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors in terminal output")
    p.add_argument("--no-reasoning", action="store_true", help="Hide reasoning_content (thinking traces)")
    p.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for webfetch (UNSAFE)")
    p.add_argument("--bash-enabled", action="store_true", help="Enable 'bash' tool. Security Warning!")
    # MCP
    p.add_argument("--mcp-url", default=os.getenv("MCP_URL"), help="MCP server base URL (optional)")
    p.add_argument("--mcp-token", default=os.getenv("MCP_TOKEN"), help="Bearer token for MCP server (optional)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    opts = parse_args(argv)
    try:
        agent_loop(opts)
    except Exception as e:
        print(colorize(f"Error: {e}", Colors.RED))
        sys.exit(1)


if __name__ == "__main__":
    main()
