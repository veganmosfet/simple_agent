#!/usr/bin/env python3
"""SCADA datapoint MCP mock server implemented with the official MCP Python SDK."""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP


class StaticTokenVerifier(TokenVerifier):
    """Minimal token verifier that checks a single bearer token."""

    def __init__(self, expected_token: str):
        self._expected = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != self._expected:
            return None
        return AccessToken(
            token=token,
            client_id="scada-mock-client",
            scopes=["dp:read", "dp:write"],
            expires_at=None,
        )


@dataclass(slots=True)
class DataPoint:
    """Represents a single SCADA datapoint."""

    name: str
    dtype: str
    value: Union[bool, float]

    def __post_init__(self) -> None:
        if not self.name.startswith("SYS:"):
            raise ValueError("Datapoint names must start with 'SYS:'")
        if self.dtype not in {"bool", "float"}:
            raise ValueError(f"Unsupported datapoint type: {self.dtype}")
        self.value = self._coerce_value(self.value)

    def _coerce_value(self, value: Union[bool, float]) -> Union[bool, float]:
        if self.dtype == "bool":
            if isinstance(value, bool):
                return value
            raise TypeError(f"{self.name} expects a boolean value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, bool):
            # Avoid silently accepting bool for float points.
            raise TypeError(f"{self.name} expects a numeric value")
        raise TypeError(f"{self.name} expects a numeric value")

    def set_value(self, value: Union[bool, float]) -> Union[bool, float]:
        self.value = self._coerce_value(value)
        return self.value

    def snapshot(self) -> Dict[str, Union[str, bool, float]]:
        return {"name": self.name, "type": self.dtype, "value": self.value}


class DataPointStore:
    """Thread-safe in-memory datapoint registry."""

    def __init__(self, initial: Dict[str, Tuple[str, Union[bool, float]]]):
        self._lock = threading.RLock()
        self._points: Dict[str, DataPoint] = {
            name: DataPoint(name, dtype, value) for name, (dtype, value) in initial.items()
        }

    def list_points(self) -> List[Dict[str, Union[str, bool, float]]]:
        with self._lock:
            return [dp.snapshot() for dp in self._points.values()]

    def get_point(self, name: str) -> DataPoint:
        with self._lock:
            if name not in self._points:
                raise KeyError(f"Unknown datapoint: {name}")
            return self._points[name]

    def describe(self, name: str) -> Dict[str, str]:
        dp = self.get_point(name)
        return {"name": dp.name, "type": dp.dtype}

    def get_value(self, name: str) -> Dict[str, Union[str, bool, float]]:
        return self.get_point(name).snapshot()

    def set_value(self, name: str, value: Union[bool, float]) -> Dict[str, Union[str, bool, float]]:
        dp = self.get_point(name)
        dp.set_value(value)
        print("############")
        print("[LOG_DP_SET]", name, value)
        print("############")
        return dp.snapshot()


def build_default_store() -> DataPointStore:
    """Return a store seeded with a few SCADA-style datapoints."""
    initial = {
        "SYS:PUMP01.RUN": ("bool", False),
        "SYS:PUMP01.FLOW": ("float", 12.4),
        "SYS:TANK01.LEVEL": ("float", 55.0),
        "SYS:LINE01.ALARM": ("bool", True),
    }
    return DataPointStore(initial)


def create_server(
    *,
    host: str,
    port: int,
    log_level: str,
    token: str,
    mount_path: str,
    issuer_url: str,
    resource_url: str,
) -> FastMCP:
    """Create a FastMCP server instance with datapoint tools."""
    store = build_default_store()
    auth_settings = AuthSettings(
        issuer_url=issuer_url,
        resource_server_url=resource_url,
        required_scopes=["dp:read", "dp:write"],
    )

    server = FastMCP(
        name="SCADA Datapoint Mock",
        instructions="Use dp-list/dp-get/dp-set/dp-type tools to inspect SCADA datapoints.",
        host=host,
        port=port,
        log_level=log_level,
        token_verifier=StaticTokenVerifier(token),
        mount_path=mount_path,
        auth=auth_settings,
        stateless_http=True,
    )

    @server.tool(name="dp-list", description="List every datapoint in the SCADA project and its current value and type.")
    def dp_list() -> List[Dict[str, Union[str, bool, float]]]:
        return store.list_points()

    @server.tool(name="dp-get", description="Read the current value of a single datapoint in the SCADA project.")
    def dp_get(name: str) -> Dict[str, Union[str, bool, float]]:
        return store.get_value(name)

    @server.tool(name="dp-set", description="Set the value of a single datapoint in the SCADA project. CAUTION: This operation directly controls real industrial equipment. Use with care in production environments.")
    async def dp_set(
        name: str,
        value: Union[bool, float],
        ctx: Context | None = None,
    ) -> Dict[str, Union[str, bool, float]]:
        result = store.set_value(name, value)
        if ctx is not None:
            await ctx.log("info", f"{name} updated to {result['value']}")
        return result

    @server.tool(name="dp-type", description="Return metadata for a single datapoint (name and type) in the SCADA project.")
    def dp_type(name: str) -> Dict[str, str]:
        return store.describe(name)

    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock SCADA MCP server using the modelcontextprotocol SDK.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--log-level", default="INFO", help="Logging level for uvicorn and FastMCP (default: INFO)")
    parser.add_argument("--token", required=True, help="Bearer token expected in the Authorization header.")
    parser.add_argument("--mount-path", default="/", help="Optional mount path when hosting behind a reverse proxy.")
    parser.add_argument(
        "--transport",
        choices=("sse", "streamable-http"),
        default="streamable-http",
        help="Transport to expose (default: streamable-http to match most MCP agents; use 'sse' for raw SSE/JSON-RPC).",
    )
    parser.add_argument(
        "--issuer-url",
        default=None,
        help="OAuth issuer URL advertised to clients (defaults to http://<host>:<port>/issuer).",
    )
    parser.add_argument(
        "--resource-url",
        default=None,
        help="Resource server URL advertised in WWW-Authenticate (defaults to http://<host>:<port>).",
    )
    return parser.parse_args()


async def run_server(server: FastMCP, transport: str, mount_path: str) -> None:
    if transport == "streamable-http":
        await server.run_streamable_http_async()
        return
    await server.run_sse_async(mount_path=mount_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    issuer_url = args.issuer_url or f"http://{args.host}:{args.port}/issuer"
    resource_url = args.resource_url or f"http://{args.host}:{args.port}"
    server = create_server(
        host=args.host,
        port=args.port,
        log_level=args.log_level.upper(),
        token=args.token,
        mount_path=args.mount_path,
        issuer_url=issuer_url,
        resource_url=resource_url,
    )
    try:
        asyncio.run(run_server(server, args.transport, args.mount_path))
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")


if __name__ == "__main__":
    main()
