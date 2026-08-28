"""Coverage-policy orchestration — CRUD, evaluation, and blocking enforcement.

Persistence is delegated to
:class:`~backend.services.app_settings_service.AppSettingsService` (JSON blobs
in ``dq_app_settings``). Live table state — columns, primary keys, and the
checks currently applied — is read through the injected
:class:`~backend.services.discovery.DiscoveryService` (Unity Catalog) and
:class:`~backend.services.rules_catalog_service.RulesCatalogService`
(``dq_quality_rules``). The pure matching / evaluation logic lives in
:mod:`backend.coverage_models`; this service only wires the I/O around it.
"""

import logging
from typing import Any
from uuid import uuid4

from databricks_labs_dqx_app.backend.coverage_models import (
    COVERAGE_COLUMN_TYPES,
    COVERAGE_ENFORCEMENTS,
    CoveragePolicy,
    CoveragePolicyEvaluation,
    CoverageReport,
    CoverageWaiver,
    evaluate_policy,
)

logger = logging.getLogger(__name__)


class CoverageBlockedError(Exception):
    """Raised when a blocking coverage policy is unsatisfied for a binding.

    Carries the human-readable, multi-line detail the route surfaces (as a 409)
    plus the machine-readable report so callers can render structured UI.
    """

    def __init__(self, detail: str, report: CoverageReport) -> None:
        super().__init__(detail)
        self.detail = detail
        self.report = report


class CoveragePolicyService:
    """Read/write coverage policies + waivers and evaluate table compliance."""

    def __init__(
        self,
        app_settings: Any,
        rules_catalog: Any,
        discovery: Any,
        apply_rules: Any = None,
    ) -> None:
        self._settings = app_settings
        self._rules = rules_catalog
        self._discovery = discovery
        # Optional so unit tests can construct the service without the tier-2
        # apply link; when present it lets a policy's ``required_check`` name a
        # registry rule_id (matched by "is this rule applied to the column")
        # in addition to a raw DQX function name (matched against materialized
        # dq_quality_rules).
        self._apply_rules = apply_rules

    # ------------------------------------------------------------------
    # Policy CRUD
    # ------------------------------------------------------------------

    def list_policies(self) -> list[CoveragePolicy]:
        """Return all persisted policies, skipping any malformed row."""
        out: list[CoveragePolicy] = []
        for raw in self._settings.get_coverage_policies_raw():
            try:
                out.append(CoveragePolicy.model_validate(raw))
            except Exception as exc:
                logger.warning("Skipping malformed coverage policy %r: %s", raw, exc)
        return out

    def get_policy(self, policy_id: str) -> CoveragePolicy | None:
        for policy in self.list_policies():
            if policy.policy_id == policy_id:
                return policy
        return None

    def replace_policies(
        self,
        policies: list[CoveragePolicy],
        *,
        user_email: str | None = None,
    ) -> list[CoveragePolicy]:
        """Validate + persist the full policy list, minting ids for new rows.

        Raises:
            ValueError: a policy has a blank name, no table scope, no required
                rule, an invalid enforcement / column_type, or a duplicate
                ``policy_id``.
        """
        cleaned: list[CoveragePolicy] = []
        seen_ids: set[str] = set()
        for policy in policies:
            name = (policy.name or "").strip()
            if not name:
                raise ValueError("Coverage policy 'name' cannot be blank.")
            scopes = policy.effective_scopes()
            if not scopes:
                raise ValueError(f"Coverage policy {name!r} must have at least one non-empty table scope.")
            checks = policy.effective_checks()
            if not checks:
                raise ValueError(f"Coverage policy {name!r} must specify at least one required rule.")
            if policy.enforcement not in COVERAGE_ENFORCEMENTS:
                raise ValueError(f"Coverage policy {name!r} has invalid enforcement {policy.enforcement!r}.")
            if policy.column_type not in COVERAGE_COLUMN_TYPES:
                raise ValueError(f"Coverage policy {name!r} has invalid column type {policy.column_type!r}.")

            policy_id = (policy.policy_id or "").strip() or uuid4().hex[:16]
            if policy_id in seen_ids:
                raise ValueError(f"Duplicate coverage policy id: {policy_id}.")
            seen_ids.add(policy_id)

            cleaned.append(
                policy.model_copy(
                    update={
                        "policy_id": policy_id,
                        "name": name,
                        "description": (policy.description or "").strip(),
                        # Persist the normalized lists; keep the singular
                        # table_scope / required_check as the first entry so
                        # older readers still see one scope and one requirement.
                        "table_scopes": scopes,
                        "table_scope": scopes[0],
                        "required_checks": checks,
                        "required_check": checks[0],
                    }
                )
            )

        self._settings.save_coverage_policies_raw(
            [p.model_dump() for p in cleaned], user_email=user_email
        )
        # Dropping a policy should not leave dangling waivers pointing at it.
        self._prune_orphan_waivers(seen_ids, user_email=user_email)
        return cleaned

    # ------------------------------------------------------------------
    # Waiver CRUD
    # ------------------------------------------------------------------

    def list_waivers(self, binding_id: str | None = None) -> list[CoverageWaiver]:
        out: list[CoverageWaiver] = []
        for raw in self._settings.get_coverage_waivers_raw():
            try:
                waiver = CoverageWaiver.model_validate(raw)
            except Exception as exc:
                logger.warning("Skipping malformed coverage waiver %r: %s", raw, exc)
                continue
            if binding_id is None or waiver.binding_id == binding_id:
                out.append(waiver)
        return out

    def grant_waiver(
        self,
        binding_id: str,
        policy_id: str,
        reason: str,
        *,
        user_email: str | None = None,
    ) -> CoverageWaiver:
        """Grant (upsert) a waiver of *policy_id* for *binding_id*."""
        from datetime import datetime, timezone

        waivers = self.list_waivers()
        waivers = [w for w in waivers if not (w.binding_id == binding_id and w.policy_id == policy_id)]
        waiver = CoverageWaiver(
            binding_id=binding_id,
            policy_id=policy_id,
            reason=(reason or "").strip(),
            granted_by=user_email,
            granted_at=datetime.now(timezone.utc).isoformat(),
        )
        waivers.append(waiver)
        self._settings.save_coverage_waivers_raw([w.model_dump() for w in waivers], user_email=user_email)
        return waiver

    def revoke_waiver(self, binding_id: str, policy_id: str, *, user_email: str | None = None) -> bool:
        """Remove a waiver. Returns True when one was removed."""
        waivers = self.list_waivers()
        remaining = [w for w in waivers if not (w.binding_id == binding_id and w.policy_id == policy_id)]
        if len(remaining) == len(waivers):
            return False
        self._settings.save_coverage_waivers_raw([w.model_dump() for w in remaining], user_email=user_email)
        return True

    def _prune_orphan_waivers(self, live_policy_ids: set[str], *, user_email: str | None) -> None:
        waivers = self.list_waivers()
        kept = [w for w in waivers if w.policy_id in live_policy_ids]
        if len(kept) != len(waivers):
            self._settings.save_coverage_waivers_raw([w.model_dump() for w in kept], user_email=user_email)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def build_report(self, binding_id: str, table_fqn: str) -> CoverageReport:
        """Evaluate every enabled, in-scope policy against *table_fqn*.

        Reads live columns + primary keys from Unity Catalog and the checks
        currently applied to the table from ``dq_quality_rules``. Any read
        failure degrades gracefully: unresolved columns yield an empty report,
        unresolved primary keys flip ``primary_key_unknown`` so a
        ``primary_key_only`` policy is not silently treated as satisfied.
        """
        columns, pk_cols, pk_known = self._read_columns_and_pk(table_fqn)
        covered = self._read_covered_checks(table_fqn)
        covered |= self._read_applied_rule_coverage(binding_id)
        waivers = {w.policy_id: w for w in self.list_waivers(binding_id)}

        col_tuples = [(c[0], c[1], c[0] in pk_cols) for c in columns]

        evaluations: list[CoveragePolicyEvaluation] = []
        blocking_unsatisfied: list[str] = []
        for policy in self.list_policies():
            if not policy.enabled:
                continue
            if not policy.matches_table(table_fqn):
                continue
            evaluation = evaluate_policy(policy, col_tuples, covered)
            waiver = waivers.get(policy.policy_id)
            if waiver is not None:
                evaluation.waived = True
                evaluation.waiver_reason = waiver.reason
            evaluations.append(evaluation)
            if (
                policy.enforcement == "blocking"
                and not evaluation.satisfied
                and not evaluation.waived
            ):
                blocking_unsatisfied.append(policy.policy_id)

        return CoverageReport(
            binding_id=binding_id,
            table_fqn=table_fqn,
            evaluations=evaluations,
            blocking_unsatisfied=blocking_unsatisfied,
            compliant=not blocking_unsatisfied,
            primary_key_unknown=not pk_known,
        )

    def enforce_blocking(self, binding_id: str, table_fqn: str) -> None:
        """Raise :class:`CoverageBlockedError` if a blocking policy is unmet.

        No-op when every blocking policy is satisfied or waived (or there are no
        blocking policies in scope).
        """
        report = self.build_report(binding_id, table_fqn)
        if report.compliant:
            return
        lines = [f"Cannot approve {table_fqn}: {len(report.blocking_unsatisfied)} coverage policy(ies) unsatisfied."]
        for evaluation in report.evaluations:
            if evaluation.policy_id in report.blocking_unsatisfied:
                cols = ", ".join(evaluation.missing_columns[:20]) or "(matching columns)"
                checks = evaluation.required_checks or [evaluation.required_check]
                required = ", ".join(f"'{c}'" for c in checks)
                lines.append(f"  • {evaluation.name}: {required} missing on: {cols}")
        raise CoverageBlockedError("\n".join(lines), report)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _read_columns_and_pk(self, table_fqn: str) -> tuple[list[tuple[str, str]], set[str], bool]:
        """Return ``([(name, type_name)...], pk_columns, pk_known)``."""
        columns: list[tuple[str, str]] = []
        try:
            catalog, schema, table = table_fqn.split(".", 2)
            cols = self._discovery.get_table_columns(catalog, schema, table)
            columns = [(c.name, c.type_name) for c in cols if c.name]
        except Exception:
            logger.warning("Coverage: failed to read columns for %s", table_fqn, exc_info=True)
            return [], set(), False

        pk_cols: set[str] = set()
        pk_known = False
        try:
            pk_cols = self._discovery.get_primary_key_columns(table_fqn)
            pk_known = True
        except Exception:
            logger.warning("Coverage: failed to read primary keys for %s", table_fqn, exc_info=True)
        return columns, pk_cols, pk_known

    def _read_covered_checks(self, table_fqn: str) -> set[tuple[str, str]]:
        """Return ``(check_function, column_name_lower)`` pairs applied to the table.

        Sourced from every non-rejected ``dq_quality_rules`` row for the table —
        the runner-facing truth of what is (or is about to be) enforced.
        """
        covered: set[tuple[str, str]] = set()
        try:
            entries = self._rules.list_rules_for_table(table_fqn)
        except Exception:
            logger.warning("Coverage: failed to list applied checks for %s", table_fqn, exc_info=True)
            return covered

        for entry in entries:
            if getattr(entry, "status", None) == "rejected":
                continue
            for check in getattr(entry, "checks", []) or []:
                self._collect_check_columns(check, covered)
        return covered

    def _read_applied_rule_coverage(self, binding_id: str) -> set[tuple[str, str]]:
        """Return ``(rule_id, column_name_lower)`` pairs applied to *binding_id*.

        Lets a coverage policy require a specific REGISTRY RULE (by rule_id) on
        every matching column — "is this authored rule applied here" — rather
        than only a raw DQX check function. Sourced from ``dq_applied_rules``
        (the tier-2 apply link), so it reflects the rules an author has bound to
        the table's columns, mode-agnostically. Degrades to an empty set when no
        apply service is wired or the read fails.
        """
        if self._apply_rules is None:
            return set()
        covered: set[tuple[str, str]] = set()
        try:
            applied = self._apply_rules.list_applied(binding_id)
        except Exception:
            logger.warning("Coverage: failed to list applied rules for %s", binding_id, exc_info=True)
            return covered
        for rule in applied:
            rule_id = str(getattr(rule, "rule_id", "") or "")
            if not rule_id:
                continue
            for group in getattr(rule, "column_mapping", []) or []:
                if not isinstance(group, dict):
                    continue
                for column in group.values():
                    if isinstance(column, str) and column.strip():
                        covered.add((rule_id, column.strip().lower()))
        return covered

    @staticmethod
    def _collect_check_columns(check: dict, covered: set[tuple[str, str]]) -> None:
        inner = check.get("check", check) if isinstance(check, dict) else {}
        if not isinstance(inner, dict):
            return
        fn = str(inner.get("function", "") or "")
        if not fn:
            return
        args = inner.get("arguments", {}) or {}
        if not isinstance(args, dict):
            return
        cols: list[str] = []
        for key in ("column", "col_name"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                cols.append(value)
        multi = args.get("columns")
        if isinstance(multi, list):
            cols.extend([c for c in multi if isinstance(c, str) and c.strip()])
        for col in cols:
            covered.add((fn, col.strip().lower()))
