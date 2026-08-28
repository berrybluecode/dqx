"""Unit tests for coverage policies — pure matching + service orchestration.

Covers the pure helpers in ``backend.coverage_models`` (scope/type matching and
single-policy evaluation) and the ``CoveragePolicyService`` orchestration (CRUD
validation + id minting, waivers, and the per-binding compliance report) using
in-memory fakes for settings/discovery/rules-catalog so no I/O is required.
"""

import pytest

from databricks_labs_dqx_app.backend.coverage_models import (
    CoveragePolicy,
    column_type_matches,
    evaluate_policy,
    scope_matches,
)
from databricks_labs_dqx_app.backend.services.coverage_policy_service import (
    CoverageBlockedError,
    CoveragePolicyService,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestScopeMatches:
    @pytest.mark.parametrize(
        ("scope", "fqn", "expected"),
        [
            ("main.finance.*", "main.finance.orders", True),
            ("main.finance.*", "main.finance.sub.orders", True),
            ("main.finance.*", "main.sales.orders", False),
            ("main.finance.orders", "main.finance.orders", True),
            ("main.finance.orders", "main.finance.other", False),
            ("*", "anything.at.all", True),
            ("MAIN.FINANCE.*", "main.finance.orders", True),  # case-insensitive
            ("", "main.finance.orders", False),  # blank scope is inert
            # wildcard-free scope = dot-bounded catalog/schema prefix
            ("samples", "samples.bakehouse.reviews", True),  # bare catalog prefix
            ("samples", "samples", True),  # exact catalog
            ("main.finance", "main.finance.orders", True),  # schema prefix
            ("main.finance", "main.finance", True),  # exact schema
            ("main.fin", "main.finance.orders", False),  # not a dot boundary
            ("SAMPLES", "samples.bakehouse.reviews", True),  # case-insensitive prefix
            ("samples", "other.samples.t", False),  # prefix must be a real prefix
        ],
    )
    def test_scope(self, scope, fqn, expected):
        assert scope_matches(scope, fqn) is expected


class TestColumnTypeMatches:
    @pytest.mark.parametrize(
        ("selector", "type_name", "expected"),
        [
            ("timestamp", "TIMESTAMP", True),
            ("timestamp", "TIMESTAMP_NTZ", False),
            ("timestamp_ntz", "TIMESTAMP_NTZ", True),
            ("date", "DATE", True),
            ("decimal", "DECIMAL(10,2)", True),
            ("decimal", "BIGINT", False),
            ("numeric", "DECIMAL(10,2)", True),
            ("numeric", "BIGINT", True),
            ("numeric", "STRING", False),
            ("boolean", "BOOLEAN", True),
            ("string", "STRING", True),
            ("string", "VARCHAR(20)", True),
            ("any", "STRUCT<a:INT>", True),
        ],
    )
    def test_type(self, selector, type_name, expected):
        assert column_type_matches(selector, type_name) is expected


class TestEvaluatePolicy:
    def _policy(self, **kw):
        base = dict(
            policy_id="p1",
            name="P1",
            table_scope="main.finance.*",
            column_type="timestamp",
            required_check="is_not_null",
        )
        base.update(kw)
        return CoveragePolicy(**base)

    def test_satisfied_when_all_matched_columns_covered(self):
        policy = self._policy()
        cols = [("created_at", "TIMESTAMP", False), ("amount", "DECIMAL(10,2)", False)]
        covered = {("is_not_null", "created_at")}
        result = evaluate_policy(policy, cols, covered)
        assert result.matched_columns == ["created_at"]
        assert result.satisfied is True
        assert result.missing_columns == []

    def test_unsatisfied_lists_missing(self):
        policy = self._policy()
        cols = [("created_at", "TIMESTAMP", False), ("updated_at", "TIMESTAMP", False)]
        covered = {("is_not_null", "created_at")}
        result = evaluate_policy(policy, cols, covered)
        assert set(result.matched_columns) == {"created_at", "updated_at"}
        assert result.covered_columns == ["created_at"]
        assert result.missing_columns == ["updated_at"]
        assert result.satisfied is False

    def test_primary_key_only_narrows_scope(self):
        policy = self._policy(column_type="string", required_check="is_unique", primary_key_only=True)
        cols = [("id", "STRING", True), ("name", "STRING", False)]
        covered = set()
        result = evaluate_policy(policy, cols, covered)
        assert result.matched_columns == ["id"]  # 'name' is not a PK
        assert result.missing_columns == ["id"]

    def test_no_matched_columns_is_trivially_satisfied(self):
        policy = self._policy(column_type="boolean")
        cols = [("created_at", "TIMESTAMP", False)]
        result = evaluate_policy(policy, cols, set())
        assert result.matched_columns == []
        assert result.satisfied is True

    # -- multiple required rules (required_checks list) --------------------

    def test_multiple_checks_all_present_is_satisfied(self):
        policy = self._policy(required_check="", required_checks=["is_not_null", "is_in_range"])
        cols = [("created_at", "TIMESTAMP", False)]
        covered = {("is_not_null", "created_at"), ("is_in_range", "created_at")}
        result = evaluate_policy(policy, cols, covered)
        assert result.required_checks == ["is_not_null", "is_in_range"]
        assert result.required_check == "is_not_null"  # back-compat first entry
        assert result.covered_columns == ["created_at"]
        assert result.satisfied is True

    def test_multiple_checks_one_missing_is_unsatisfied(self):
        policy = self._policy(required_check="", required_checks=["is_not_null", "is_in_range"])
        cols = [("created_at", "TIMESTAMP", False)]
        covered = {("is_not_null", "created_at")}  # is_in_range absent
        result = evaluate_policy(policy, cols, covered)
        assert result.missing_columns == ["created_at"]
        assert result.satisfied is False

    def test_required_checks_falls_back_to_legacy_single(self):
        # No required_checks list → the legacy required_check is the sole rule.
        policy = self._policy(required_check="is_not_null")
        assert policy.effective_checks() == ["is_not_null"]

    def test_effective_checks_dedupes_preserving_order(self):
        policy = self._policy(
            required_check="", required_checks=["is_not_null", "is_in_range", "is_not_null", "  "]
        )
        assert policy.effective_checks() == ["is_not_null", "is_in_range"]


class TestEffectiveScopesAndMatchesTable:
    def _policy(self, **kw):
        base = dict(policy_id="p1", name="P1", column_type="any", required_check="is_not_null")
        base.update(kw)
        return CoveragePolicy(**base)

    def test_scopes_falls_back_to_legacy_single(self):
        # No table_scopes list → the legacy table_scope is the sole scope.
        policy = self._policy(table_scope="main.finance.*")
        assert policy.effective_scopes() == ["main.finance.*"]

    def test_effective_scopes_dedupes_preserving_order(self):
        policy = self._policy(
            table_scope="", table_scopes=["main.finance.*", "prod.sales", "main.finance.*", "  "]
        )
        assert policy.effective_scopes() == ["main.finance.*", "prod.sales"]

    def test_matches_table_when_any_scope_matches(self):
        policy = self._policy(table_scope="", table_scopes=["main.finance.*", "prod.sales.orders"])
        assert policy.matches_table("main.finance.orders") is True  # first scope (glob)
        assert policy.matches_table("prod.sales.orders") is True  # second scope (exact)
        assert policy.matches_table("other.raw.events") is False  # neither

    def test_matches_table_uses_legacy_scope_when_list_empty(self):
        policy = self._policy(table_scope="samples")  # dot-bounded prefix
        assert policy.matches_table("samples.bakehouse.reviews") is True
        assert policy.matches_table("sampled.x.y") is False


# ---------------------------------------------------------------------------
# Service fakes
# ---------------------------------------------------------------------------


class _FakeSettings:
    def __init__(self):
        self.store: dict = {}

    def get_coverage_policies_raw(self):
        return list(self.store.get("policies", []))

    def save_coverage_policies_raw(self, policies, *, user_email=None):
        self.store["policies"] = policies

    def get_coverage_waivers_raw(self):
        return list(self.store.get("waivers", []))

    def save_coverage_waivers_raw(self, waivers, *, user_email=None):
        self.store["waivers"] = waivers


class _FakeColumn:
    def __init__(self, name, type_name):
        self.name = name
        self.type_name = type_name


class _FakeDiscovery:
    def __init__(self, columns, pk=None, raise_pk=False):
        self._columns = columns
        self._pk = set(pk or set())
        self._raise_pk = raise_pk

    def get_table_columns(self, catalog, schema, table):
        return self._columns

    def get_primary_key_columns(self, table_fqn):
        if self._raise_pk:
            raise RuntimeError("no perms")
        return set(self._pk)


class _FakeEntry:
    def __init__(self, checks, status="approved"):
        self.checks = checks
        self.status = status


class _FakeRules:
    def __init__(self, entries):
        self._entries = entries

    def list_rules_for_table(self, table_fqn):
        return self._entries


class _FakeAppliedRule:
    def __init__(self, rule_id, column_mapping):
        self.rule_id = rule_id
        self.column_mapping = column_mapping


class _FakeApplyRules:
    def __init__(self, applied):
        self._applied = applied

    def list_applied(self, binding_id):
        return self._applied


def _check(fn, column=None, columns=None):
    args: dict = {}
    if column is not None:
        args["column"] = column
    if columns is not None:
        args["columns"] = columns
    return {"check": {"function": fn, "arguments": args}, "name": f"{fn}_check"}


def _make_service(policies=None, waivers=None, columns=None, pk=None, entries=None, raise_pk=False, applied=None):
    settings = _FakeSettings()
    if policies is not None:
        settings.store["policies"] = policies
    if waivers is not None:
        settings.store["waivers"] = waivers
    discovery = _FakeDiscovery(columns or [], pk=pk, raise_pk=raise_pk)
    rules = _FakeRules(entries or [])
    apply_rules = _FakeApplyRules(applied) if applied is not None else None
    return (
        CoveragePolicyService(
            app_settings=settings, rules_catalog=rules, discovery=discovery, apply_rules=apply_rules
        ),
        settings,
    )


# ---------------------------------------------------------------------------
# Service — CRUD
# ---------------------------------------------------------------------------


class TestPolicyCrud:
    def test_replace_mints_ids_and_persists(self):
        svc, settings = _make_service()
        saved = svc.replace_policies(
            [
                CoveragePolicy(
                    name="Finance timestamps",
                    table_scope="main.finance.*",
                    column_type="timestamp",
                    required_check="is_not_null",
                )
            ],
            user_email="admin@x.com",
        )
        assert len(saved) == 1
        assert saved[0].policy_id  # minted
        assert settings.store["policies"][0]["name"] == "Finance timestamps"

    def test_replace_rejects_blank_name(self):
        svc, _ = _make_service()
        with pytest.raises(ValueError, match="name"):
            svc.replace_policies(
                [CoveragePolicy(name="  ", table_scope="a.b.*", column_type="any", required_check="is_not_null")]
            )

    def test_replace_rejects_blank_scope(self):
        svc, _ = _make_service()
        with pytest.raises(ValueError, match="table scope"):
            svc.replace_policies(
                [CoveragePolicy(name="P", table_scope="", column_type="any", required_check="is_not_null")]
            )

    def test_replace_persists_multiple_scopes_and_syncs_legacy(self):
        svc, settings = _make_service()
        saved = svc.replace_policies(
            [
                CoveragePolicy(
                    name="Multi",
                    table_scopes=["main.finance.*", "prod.sales.orders", "main.finance.*"],
                    column_type="any",
                    required_check="is_not_null",
                )
            ]
        )
        # De-duplicated, order preserved; legacy singular kept as first entry.
        assert saved[0].table_scopes == ["main.finance.*", "prod.sales.orders"]
        assert saved[0].table_scope == "main.finance.*"
        stored = settings.store["policies"][0]
        assert stored["table_scopes"] == ["main.finance.*", "prod.sales.orders"]
        assert stored["table_scope"] == "main.finance.*"

    def test_replace_rejects_empty_scope_list(self):
        svc, _ = _make_service()
        with pytest.raises(ValueError, match="table scope"):
            svc.replace_policies(
                [CoveragePolicy(name="P", table_scope="", table_scopes=[], column_type="any", required_check="is_not_null")]
            )

    def test_dropping_policy_prunes_waivers(self):
        svc, settings = _make_service(
            policies=[{"policy_id": "p1", "name": "P1", "table_scope": "a.b.*", "column_type": "any", "required_check": "is_not_null"}],
            waivers=[{"binding_id": "b1", "policy_id": "p1", "reason": "ok"}],
        )
        # Replace with a different policy (p1 dropped) → its waiver is pruned.
        svc.replace_policies(
            [CoveragePolicy(policy_id="p2", name="P2", table_scope="a.b.*", column_type="any", required_check="is_not_null")]
        )
        assert settings.store["waivers"] == []


class TestWaivers:
    def test_grant_and_revoke(self):
        svc, _ = _make_service()
        svc.grant_waiver("b1", "p1", "temporary exemption", user_email="admin@x.com")
        waivers = svc.list_waivers("b1")
        assert len(waivers) == 1
        assert waivers[0].reason == "temporary exemption"
        assert svc.revoke_waiver("b1", "p1") is True
        assert svc.list_waivers("b1") == []

    def test_grant_is_idempotent_upsert(self):
        svc, _ = _make_service()
        svc.grant_waiver("b1", "p1", "first")
        svc.grant_waiver("b1", "p1", "second")
        waivers = svc.list_waivers("b1")
        assert len(waivers) == 1
        assert waivers[0].reason == "second"


# ---------------------------------------------------------------------------
# Service — report + enforcement
# ---------------------------------------------------------------------------


class TestReportAndEnforcement:
    _POLICY = {
        "policy_id": "p1",
        "name": "Finance timestamps not-null",
        "enabled": True,
        "enforcement": "blocking",
        "table_scope": "main.finance.*",
        "column_type": "timestamp",
        "required_check": "is_not_null",
    }

    def test_blocking_unsatisfied_when_missing(self):
        svc, _ = _make_service(
            policies=[self._POLICY],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.compliant is False
        assert report.blocking_unsatisfied == ["p1"]
        with pytest.raises(CoverageBlockedError):
            svc.enforce_blocking("b1", "main.finance.orders")

    def test_satisfied_when_check_applied(self):
        svc, _ = _make_service(
            policies=[self._POLICY],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[_FakeEntry([_check("is_not_null", column="created_at")])],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.compliant is True
        assert report.blocking_unsatisfied == []
        svc.enforce_blocking("b1", "main.finance.orders")  # no raise

    def test_multiple_required_checks_all_needed_to_satisfy(self):
        multi = {
            **self._POLICY,
            "required_check": "",
            "required_checks": ["is_not_null", "is_in_range"],
        }
        # Only one of the two required checks is applied → still blocking.
        svc, _ = _make_service(
            policies=[multi],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[_FakeEntry([_check("is_not_null", column="created_at")])],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.blocking_unsatisfied == ["p1"]
        assert report.evaluations[0].required_checks == ["is_not_null", "is_in_range"]
        with pytest.raises(CoverageBlockedError):
            svc.enforce_blocking("b1", "main.finance.orders")

    def test_multiple_required_checks_satisfied_when_all_applied(self):
        multi = {
            **self._POLICY,
            "required_check": "",
            "required_checks": ["is_not_null", "is_in_range"],
        }
        svc, _ = _make_service(
            policies=[multi],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[
                _FakeEntry(
                    [
                        _check("is_not_null", column="created_at"),
                        _check("is_in_range", column="created_at"),
                    ]
                )
            ],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.compliant is True
        svc.enforce_blocking("b1", "main.finance.orders")  # no raise

    def test_rejected_checks_do_not_count_as_coverage(self):
        svc, _ = _make_service(
            policies=[self._POLICY],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[_FakeEntry([_check("is_not_null", column="created_at")], status="rejected")],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.blocking_unsatisfied == ["p1"]

    def test_waiver_suppresses_blocking(self):
        svc, _ = _make_service(
            policies=[self._POLICY],
            waivers=[{"binding_id": "b1", "policy_id": "p1", "reason": "signed off"}],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.compliant is True
        evaluation = report.evaluations[0]
        assert evaluation.waived is True
        assert evaluation.waiver_reason == "signed off"
        svc.enforce_blocking("b1", "main.finance.orders")  # no raise

    def test_advisory_never_blocks(self):
        advisory = {**self._POLICY, "enforcement": "advisory"}
        svc, _ = _make_service(
            policies=[advisory],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.compliant is True  # advisory is surfaced but never blocks
        assert report.evaluations[0].satisfied is False
        assert report.blocking_unsatisfied == []

    def test_disabled_policy_ignored(self):
        disabled = {**self._POLICY, "enabled": False}
        svc, _ = _make_service(
            policies=[disabled],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.evaluations == []

    def test_out_of_scope_policy_ignored(self):
        svc, _ = _make_service(
            policies=[self._POLICY],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[],
        )
        report = svc.build_report("b1", "main.sales.orders")  # not main.finance.*
        assert report.evaluations == []

    def test_multi_scope_policy_matches_second_scope(self):
        # A table under the SECOND scope must still be evaluated by the policy.
        multi = {
            **self._POLICY,
            "table_scope": "main.finance.*",
            "table_scopes": ["main.finance.*", "prod.sales.*"],
        }
        svc, _ = _make_service(
            policies=[multi],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[],
        )
        report = svc.build_report("b1", "prod.sales.orders")  # matches the 2nd scope
        assert len(report.evaluations) == 1
        assert report.blocking_unsatisfied == ["p1"]

    def test_multi_scope_policy_ignores_table_outside_all_scopes(self):
        multi = {
            **self._POLICY,
            "table_scope": "main.finance.*",
            "table_scopes": ["main.finance.*", "prod.sales.*"],
        }
        svc, _ = _make_service(
            policies=[multi],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            entries=[],
        )
        report = svc.build_report("b1", "other.raw.events")
        assert report.evaluations == []

    def test_primary_key_unknown_flag(self):
        pk_policy = {
            "policy_id": "pk1",
            "name": "PK unique",
            "enforcement": "advisory",
            "table_scope": "main.finance.*",
            "column_type": "string",
            "primary_key_only": True,
            "required_check": "is_unique",
        }
        svc, _ = _make_service(
            policies=[pk_policy],
            columns=[_FakeColumn("id", "STRING")],
            raise_pk=True,
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.primary_key_unknown is True

    def test_is_unique_columns_list_counts_as_coverage(self):
        pk_policy = {
            "policy_id": "pk1",
            "name": "PK unique",
            "enforcement": "blocking",
            "table_scope": "main.finance.*",
            "column_type": "string",
            "primary_key_only": True,
            "required_check": "is_unique",
        }
        svc, _ = _make_service(
            policies=[pk_policy],
            columns=[_FakeColumn("id", "STRING")],
            pk={"id"},
            entries=[_FakeEntry([_check("is_unique", columns=["id"])])],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.compliant is True

    # -- required_check as a REGISTRY RULE id (matched via dq_applied_rules) ----

    _RULE_POLICY = {
        "policy_id": "rp1",
        "name": "Finance timestamps carry rule R",
        "enabled": True,
        "enforcement": "blocking",
        "table_scope": "main.finance.*",
        "column_type": "timestamp",
        "required_check": "ruleabc123",  # a registry rule_id, not a function name
    }

    def test_satisfied_when_registry_rule_applied_to_column(self):
        svc, _ = _make_service(
            policies=[self._RULE_POLICY],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            applied=[_FakeAppliedRule("ruleabc123", [{"column": "created_at"}])],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.compliant is True
        assert report.evaluations[0].satisfied is True
        svc.enforce_blocking("b1", "main.finance.orders")  # no raise

    def test_unsatisfied_when_registry_rule_applied_to_other_column(self):
        svc, _ = _make_service(
            policies=[self._RULE_POLICY],
            columns=[_FakeColumn("created_at", "TIMESTAMP"), _FakeColumn("updated_at", "TIMESTAMP")],
            applied=[_FakeAppliedRule("ruleabc123", [{"column": "created_at"}])],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.evaluations[0].missing_columns == ["updated_at"]
        assert report.blocking_unsatisfied == ["rp1"]

    def test_unsatisfied_when_different_registry_rule_applied(self):
        svc, _ = _make_service(
            policies=[self._RULE_POLICY],
            columns=[_FakeColumn("created_at", "TIMESTAMP")],
            applied=[_FakeAppliedRule("some_other_rule", [{"column": "created_at"}])],
        )
        report = svc.build_report("b1", "main.finance.orders")
        assert report.blocking_unsatisfied == ["rp1"]
