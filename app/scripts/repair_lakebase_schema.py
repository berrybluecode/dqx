"""One-shot repair for stale Lakebase deployments (kaizen-dev, etc.).

Background
----------
:mod:`backend.migrations.postgres` records the initial baseline as v1 and
warns in its docstring:

    An existing deployment does not pick up an edited baseline. Its
    ``dq_migrations`` table already records v1 as applied, so the runner
    skips it and the old columns stay.

Which is exactly what bit kaizen-dev after commit ``af6588c1`` folded the
newer schedule/owner/rationale columns into v1: the workspace's
``dq_monitored_tables`` and ``dq_data_products`` tables were created
under the pre-collapse shape, so the app now fails with
``UndefinedColumn: column mt.schedule_sample_size does not exist``.

What this script does
---------------------
Runs ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` for every column on
those two tables that the current baseline expects. Idempotent — Postgres
silently skips columns that already exist — so it is safe to run against
a fresh workspace, against kaizen-dev, or twice in a row.

**Does NOT touch data, does NOT drop/recreate constraints, and does NOT
edit the ``dq_migrations`` meta table.** If we later want to run this on
every deploy, promote the ADD COLUMNs into a proper v2 ``PgMigration``
entry (see the ``postgres.py`` docstring) — this script is deliberately
narrow and off the migration ladder.

Usage (from ``app/``):

    uv run python scripts/repair_lakebase_schema.py -p <profile> -t <target>

Example (kaizen-dev):

    uv run python scripts/repair_lakebase_schema.py -p kaizen-dev -t kaizen-dev-v2
"""

import argparse
import json
import subprocess
import sys


# Every non-PK, non-required-at-create-time column the current baseline
# declares on ``dq_monitored_tables``. Ordered to match ``postgres.py`` so
# a future baseline edit is easy to reconcile line by line. The two ``NOT
# NULL DEFAULT`` columns are safe to add on non-empty tables — Postgres
# backfills existing rows with the default in a single statement.
_MT_COLUMNS: list[tuple[str, str]] = [
    ("owner", "TEXT"),
    ("owner_display_name", "TEXT"),
    ("version", "INTEGER NOT NULL DEFAULT 0"),
    ("schedule_cron", "TEXT"),
    ("schedule_tz", "TEXT"),
    ("schedule_kind", "TEXT NOT NULL DEFAULT 'dq_only'"),
    ("schedule_sample_size", "INTEGER"),
    ("last_profiled_at", "TIMESTAMPTZ"),
    ("last_run_at", "TIMESTAMPTZ"),
    ("pending_rationale", "TEXT"),
    ("last_decision_rationale", "TEXT"),
    ("created_by", "TEXT"),
    ("created_at", "TIMESTAMPTZ"),
    ("updated_by", "TEXT"),
    ("updated_at", "TIMESTAMPTZ"),
]

_DP_COLUMNS: list[tuple[str, str]] = [
    ("description", "TEXT"),
    ("owner", "TEXT"),
    ("owner_display_name", "TEXT"),
    ("schedule_cron", "TEXT"),
    ("schedule_tz", "TEXT"),
    ("schedule_kind", "TEXT NOT NULL DEFAULT 'dq_only'"),
    ("schedule_sample_size", "INTEGER"),
    ("version", "INTEGER NOT NULL DEFAULT 0"),
    ("pending_rationale", "TEXT"),
    ("last_decision_rationale", "TEXT"),
    ("created_by", "TEXT"),
    ("created_at", "TIMESTAMPTZ"),
    ("updated_by", "TEXT"),
    ("updated_at", "TIMESTAMPTZ"),
]


def _bundle_json(profile: str, target: str) -> dict:
    proc = subprocess.run(
        ["databricks", "-p", profile, "bundle", "validate", "-t", target, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print("ERROR: databricks bundle validate failed:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(1)
    return json.loads(proc.stdout)


def _workspace_host(profile: str) -> str:
    proc = subprocess.run(
        ["databricks", "-p", profile, "auth", "describe", "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print("ERROR: databricks auth describe failed:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(1)
    details = json.loads(proc.stdout).get("details", {})
    host = details.get("host") or details.get("configuration", {}).get("host", {}).get("value")
    if not host:
        print("ERROR: could not resolve workspace host.", file=sys.stderr)
        raise SystemExit(1)
    return host


def _add_column_statements(schema: str, table: str, columns: list[tuple[str, str]]) -> list[str]:
    return [f'ALTER TABLE "{schema}"."{table}" ADD COLUMN IF NOT EXISTS "{name}" {ddl};' for name, ddl in columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile stale Lakebase columns for the two crashing tables.")
    parser.add_argument("-p", "--profile", required=True, help="Databricks CLI profile")
    parser.add_argument("-t", "--target", required=True, help="Bundle target name")
    args = parser.parse_args()

    bundle = _bundle_json(args.profile, args.target)
    variables = bundle.get("variables", {})

    def _var(name: str, default: str = "") -> str:
        node = variables.get(name, {})
        return str(node.get("value") or node.get("default") or default)

    endpoint = _var("lakebase_endpoint")
    if not endpoint or endpoint == "-":
        print("==> No Lakebase endpoint configured (Delta-only mode). Nothing to repair.")
        return
    database = _var("lakebase_database_name", "databricks_postgres")
    schema = _var("lakebase_schema_name", "dqx_studio")

    from databricks.sdk import WorkspaceClient

    from databricks_labs_dqx_app.backend.pg_cursor_helpers import run_trusted_sql
    from databricks_labs_dqx_app.backend.pg_executor import build_pg_executor

    ws = WorkspaceClient(profile=args.profile, host=_workspace_host(args.profile))
    pg = build_pg_executor(ws, endpoint=endpoint, database=database, schema=schema)
    try:
        statements = _add_column_statements(schema, "dq_monitored_tables", _MT_COLUMNS)
        statements += _add_column_statements(schema, "dq_data_products", _DP_COLUMNS)
        with pg.connection() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    print(f"  SQL: {statement}")
                    run_trusted_sql(cur, statement)
            conn.commit()
        print(f"==> Reconciled {len(statements)} column(s) on schema={schema}.")
    finally:
        pg.close()


if __name__ == "__main__":
    main()
