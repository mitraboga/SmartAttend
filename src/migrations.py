import argparse
import json

from src.database import (
    CURRENT_SCHEMA_NAME,
    CURRENT_SCHEMA_VERSION,
    configured_database_url,
    current_schema_version,
    database_backend,
    init_database,
    list_schema_migrations,
)


def _status_payload() -> dict:
    return {
        "backend": database_backend(),
        "database_url_configured": bool(configured_database_url()),
        "current_version": current_schema_version(),
        "target_version": CURRENT_SCHEMA_VERSION,
        "target_name": CURRENT_SCHEMA_NAME,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartAttend schema migration utility.")
    parser.add_argument(
        "command",
        nargs="?",
        default="current",
        choices=["current", "history", "upgrade"],
        help="Migration command to run.",
    )
    args = parser.parse_args()

    if args.command == "upgrade":
        init_database()
        print(json.dumps(_status_payload(), indent=2))
        return 0

    if args.command == "history":
        init_database()
        print(json.dumps(list_schema_migrations(), indent=2))
        return 0

    print(json.dumps(_status_payload(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
