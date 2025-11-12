# Simple LLM Terminal Agent
 
This repository contains a single-file Python agent (`agent.py`) that runs in a terminal. It is compatible with OpenAI-style chat APIs, supports native tools (`bash`, `webfetch`, and `readfile`), and integrates with one MCP server over HTTP JSON‑RPC. Everything runs in the terminal with concise output and optional debug logs.

**Warning** 
This is a playground project developed to understand how agents work, not a production ready agent. Be very careful with the `bash` tool :-)

## Usage

- OpenAI-compatible Endpoint:
  - `uv run agent.py --model mlx-community/Qwen3-4B-Instruct-2507-4bit --api-base http://127.0.0.1:8080/v1 --api-key XXX`
    
- Azure OpenAI (API key):
  - `uv run agent.py --model-provider azure --model <deployment_name> --azure-endpoint https://<resource>.openai.azure.com --azure-api-version 2024-08-01-preview --api-key $AZURE_OPENAI_API_KEY`

- With MCP tools:
  - `uv run agent.py --model <model> --api-base <url> --mcp-url http://127.0.0.1:3000/mcp --mcp-token TOKEN`

Windows quick start
- Without Bash installed: `py agent.py --model gpt-4o-mini`
- With Git Bash or WSL available: `py agent.py --model gpt-4o-mini --bash`

Notes
- Exit with `exit` or `Ctrl+D`.
- History is saved to `~/.agent_history`.
- Prompt/history behavior:
  - On macOS with libedit, the input prompt is intentionally uncolored to avoid display glitches when navigating history.
  - With GNU readline (e.g., when running via `uv` where `gnureadline` is available), the prompt is colored safely using non-printing markers so arrows/history render correctly.
- The agent fetches MCP tools at startup via `tools/list` and exposes them to the model; calls route through `tools/call`. SSE (`text/event-stream`) responses are supported.

### Flags
- `--model <name>`: model identifier or Azure deployment name
- `--api-base <url>`: base URL for OpenAI-compatible APIs (non-Azure)
- `--api-key <key>`: API key for OpenAI or Azure
- `--model-provider <openai|azure>`: select provider (`openai` default)
- `--azure-endpoint <url>`: Azure OpenAI endpoint, e.g., `https://<resource>.openai.azure.com`
- `--azure-api-version <version>`: Azure OpenAI API version (default `2024-08-01-preview`)
- `--mcp-url <url>`: MCP JSON‑RPC endpoint (optional)
- `--mcp-token <token>`: Bearer token for MCP (optional)
- `--system-prompt <text>`: override system prompt (optional)
- `--debug-tools`: show tool calls and I/O (incl. MCP JSON‑RPC)
- `--debug-llm`: show raw LLM traffic (requests/responses)
- `--debug`: convenience flag enabling both `--debug-tools` and `--debug-llm`
- `--no-color`: disable colored terminal output
- `--bash`: enable the `bash` tool (useful on Windows without Bash). Warning! Use only in sandboxed environment!
- `--readfile-bytes <int>`: limit for `readfile` tool output (first N bytes; default 4096)

### Native Tools (exposed to the model)
- `bash({ command })`
  - Runs a shell command and returns `{ stdout, stderr, exit_code }`
  - Defaults: `timeout=30s`, `cwd=current directory`
  - Availability: disabled by default; enabled when `--bash` is provided.
  - **WARNING** Use only in sandboxed environment!
- `webfetch({ url })`
  - Fetches the URL and returns plain text extracted from HTML
  - Defaults: `user_agent="Mozilla/5.0 (Agent)", timeout=30s`
- `readfile({ filename })`
  - Reads the first `N` bytes (set via `--readfile-bytes`) of a file in the current working directory
  - Returns `{ filename, bytes_read, content }`

### MCP Tools (dynamic)
- At startup, the agent POSTs `tools/list` to `--mcp-url` and converts each tool to OpenAI-style tool definitions using their `inputSchema`.
- When the model calls one of these tools, the agent POSTs `tools/call` with `{ name, arguments }` and returns the server’s result.
- Accept headers: `application/json, text/event-stream` to support servers that stream JSON‑RPC via SSE.
- Name normalization: the agent tolerates underscore/hyphen and case variations when matching tool names (e.g., `dp_set` maps to `dp-set`).

### Tool call logging
- Every tool call prints a concise, single-line message (regardless of debug flags), e.g.:
  - `[tool] bash ls -la /tmp`
  - `[tool] webfetch https://example.com`
  - `[tool] dp_set -> dp-set {"datapoints":[...]}`
  - The arrow indicates a normalized MCP tool name used for dispatch.

### Prompting tips
- To encourage tool calls with correct parameters, ask explicitly (e.g., “Use dp-set with its schema and set SYS:TEST to 1”).
- If a model does not support tool calling, you may need a different model or endpoint.

## MCP Test with the Mockup

- A simple MCP server with SCADA mockup can be found in `MCPMOCKUP/`.
- Usage:
```
usage: scadamcpmockup.py [-h] [--host HOST] [--port PORT] [--log-level LOG_LEVEL] --token TOKEN [--mount-path MOUNT_PATH] [--transport {sse,streamable-http}] [--issuer-url ISSUER_URL]
                         [--resource-url RESOURCE_URL]
```
- Example: `uv run MCPMOCKUP/scadamcpmockup.py --token TOKEN --port 3000`
  
## Requirements
- Python >=3.12
- `openai` Python package

## File Overview
- `agent.py`: main agent (single file)
- `README.md`: this document
- `MCPMOCKUP/scadamcpmockup.py`: the MCP mockup
