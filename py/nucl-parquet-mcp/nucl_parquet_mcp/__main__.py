"""Entry point for nucl-parquet MCP server."""

from nucl_parquet_mcp.server import mcp


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
