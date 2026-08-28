"""Coverage policies — admin-managed, rule-level column-coverage requirements.

A *coverage policy* declares that every column of a given TYPE (optionally only
the table's primary key) on tables matching a SCOPE **must** carry a specific
DQX check. Two enforcement levels:

* ``advisory`` — surface missing coverage on the table page and in the
  compliance report; never blocks anything.
* ``blocking`` — a monitored-table binding cannot move to ``approved`` while a
  blocking policy is unsatisfied for its table, unless an admin waiver exists.

Column-type driven examples the model expresses:

* Every ``timestamp`` / ``timestamp_ntz`` column on ``main.finance.*`` MUST have
  a ``is_not_null`` rule.
* Every ``decimal`` column MUST have a ``is_not_less_than`` (not-negative) rule.
* Every ``date`` column MUST have a ``is_not_in_future`` rule.
* Every ``string`` primary-key column MUST have ``is_unique``.
* Every ``boolean`` column MUST have a ``is_not_null`` rule.
* Every ``numeric`` column SHOULD have a ``is_in_range`` rule (advisory).

This module is PURE (no I/O) so the scope/type matching and coverage evaluation
are trivially unit-testable. Persistence lives in
:class:`~backend.services.app_settings_service.AppSettingsService`; the
orchestration that reads live columns / applied checks / waivers lives in
:class:`~backend.services.coverage_policy_service.CoveragePolicyService`.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Enforcement levels a policy can carry.
CoverageEnforcement = Literal["advisory", "blocking"]

#: The column-type selector a policy targets. ``numeric`` is the broad family
#: (any integer/float/decimal); ``decimal`` is narrower (fixed-point money
#: columns only). ``timestamp`` and ``timestamp_ntz`` are kept distinct so a
#: policy can target one without the other.
CoverageColumnType = Literal[
    "timestamp",
    "timestamp_ntz",
    "date",
    "decimal",
    "numeric",
    "boolean",
    "string",
    "any",
]

#: Every accepted enforcement value — used to validate an incoming/stored value.
COVERAGE_ENFORCEMENTS: frozenset[str] = frozenset({"advisory", "blocking"})

#: Every accepted column-type selector value.
COVERAGE_COLUMN_TYPES: frozenset[str] = frozenset(
    {"timestamp", "timestamp_ntz", "date", "decimal", "numeric", "boolean", "string", "any"}
)

# The set of UC head types considered "numeric" — mirrors
# ``services.tag_mapping_service._TYPE_FAMILY`` numeric entries so the two
# classifiers never drift.
_NUMERIC_HEAD_TYPES: frozenset[str] = frozenset(
    {"TINYINT", "SMALLINT", "INT", "INTEGER", "BIGINT", "LONG", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"}
)
_STRING_HEAD_TYPES: frozenset[str] = frozenset({"STRING", "VARCHAR", "CHAR"})


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


def _dedup_stripped(values: list[str]) -> list[str]:
    """Strip, drop blanks, and de-duplicate *values* preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for entry in values:
        value = (entry or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


class CoveragePolicy(BaseModel):
    """One admin-managed coverage requirement (see module docstring)."""

    policy_id: str = Field(default="", description="Stable id (uuid hex); generated server-side when blank")
    name: str = Field(description="Human-readable policy name")
    description: str = Field(default="", description="Optional free-text explanation")
    enabled: bool = Field(default=True, description="Disabled policies are neither surfaced nor enforced")
    enforcement: CoverageEnforcement = Field(
        default="advisory",
        description="'advisory' (surface only) or 'blocking' (gate binding approval until satisfied/waived)",
    )
    table_scope: str = Field(
        default="",
        description=(
            "DEPRECATED single-scope form, kept for back-compat. Prefer ``table_scopes``. When "
            "``table_scopes`` is empty this one value is used as the sole scope. A '*' glob "
            "matches any run of characters (e.g. 'main.finance.*'); a wildcard-free value is a "
            "dot-bounded prefix, so 'samples' covers every table in the 'samples' catalog and "
            "'main.finance' covers that schema."
        ),
    )
    table_scopes: list[str] = Field(
        default_factory=list,
        description=(
            "The scopes the policy applies to; a table matches when it matches ANY entry (OR). "
            "Each entry has the same glob/dot-bounded-prefix semantics as ``table_scope``. When "
            "empty, ``table_scope`` is used as the single scope."
        ),
    )
    column_type: CoverageColumnType = Field(description="Which columns the policy applies to")
    primary_key_only: bool = Field(
        default=False,
        description="When true, only columns that are part of the table's PRIMARY KEY are in scope.",
    )
    required_check: str = Field(
        default="",
        description=(
            "DEPRECATED single-rule form, kept for back-compat. Prefer ``required_checks``. When "
            "``required_checks`` is empty this one value is used as the sole requirement. Identifies "
            "the rule a matching column must carry: a registry rule id (``dq_rules.rule_id``) — the "
            "column must have that authored rule applied — or a raw DQX check-function name "
            "(e.g. 'is_not_null') matched against the checks materialized on the table."
        ),
    )
    required_checks: list[str] = Field(
        default_factory=list,
        description=(
            "The rules EVERY matching column must carry (all of them). Each entry is a registry rule "
            "id (``dq_rules.rule_id``) or a raw DQX check-function name, same semantics as "
            "``required_check``. When empty, ``required_check`` is used as the single requirement."
        ),
    )

    def effective_checks(self) -> list[str]:
        """The rules this policy requires, de-duplicated in first-seen order.

        Prefers ``required_checks``; falls back to the legacy single
        ``required_check`` so older stored/inbound policies keep working.
        """
        return _dedup_stripped(self.required_checks if self.required_checks else [self.required_check])

    def effective_scopes(self) -> list[str]:
        """The scopes this policy applies to, de-duplicated in first-seen order.

        Prefers ``table_scopes``; falls back to the legacy single
        ``table_scope`` so older stored/inbound policies keep working.
        """
        return _dedup_stripped(self.table_scopes if self.table_scopes else [self.table_scope])

    def matches_table(self, table_fqn: str) -> bool:
        """Whether *table_fqn* falls under ANY of the policy's scopes."""
        return any(scope_matches(scope, table_fqn) for scope in self.effective_scopes())


class CoverageWaiver(BaseModel):
    """An admin-granted exemption of one policy for one monitored-table binding."""

    binding_id: str
    policy_id: str
    reason: str = ""
    granted_by: str | None = None
    granted_at: str | None = None


class CoveragePolicyEvaluation(BaseModel):
    """Result of evaluating one policy against one table's live state."""

    policy_id: str
    name: str
    enforcement: CoverageEnforcement
    #: First required rule — kept for back-compat with single-rule consumers.
    required_check: str
    #: Every rule the policy requires (all must be present on a matched column).
    required_checks: list[str] = Field(default_factory=list)
    column_type: CoverageColumnType
    primary_key_only: bool
    matched_columns: list[str] = Field(default_factory=list, description="Columns the policy applies to")
    covered_columns: list[str] = Field(
        default_factory=list, description="Matched columns that carry ALL required checks"
    )
    missing_columns: list[str] = Field(
        default_factory=list, description="Matched columns missing at least one required check"
    )
    satisfied: bool = Field(description="True when every matched column carries every required check")
    waived: bool = Field(default=False, description="True when an admin waiver suppresses this policy for the binding")
    waiver_reason: str | None = None


class CoverageReport(BaseModel):
    """Full per-binding compliance report across all enabled policies."""

    binding_id: str
    table_fqn: str
    evaluations: list[CoveragePolicyEvaluation] = Field(default_factory=list)
    #: Policy ids that are ``blocking``, unsatisfied, and NOT waived — the set
    #: that will refuse a binding approval.
    blocking_unsatisfied: list[str] = Field(default_factory=list)
    #: True when nothing blocking is unsatisfied (approval is allowed).
    compliant: bool = True
    #: True when the table's primary-key columns could not be resolved from UC.
    #: Callers surface this so a ``primary_key_only`` policy's "0 columns in
    #: scope" result is not silently mistaken for "fully covered".
    primary_key_unknown: bool = False


# ---------------------------------------------------------------------------
# Pure matching helpers
# ---------------------------------------------------------------------------


def head_type(type_name: str) -> str:
    """Return the upper-cased head of a UC type, dropping params/generics.

    ``"DECIMAL(10,2)"`` -> ``"DECIMAL"``; ``"ARRAY<STRING>"`` -> ``"ARRAY"``.
    """
    return (type_name or "").upper().split("(")[0].split("<")[0].strip()


def column_type_matches(selector: str, type_name: str) -> bool:
    """Whether a UC column *type_name* satisfies a policy's ``column_type``."""
    head = head_type(type_name)
    if selector == "any":
        return True
    if selector == "timestamp":
        return head == "TIMESTAMP"
    if selector == "timestamp_ntz":
        return head == "TIMESTAMP_NTZ"
    if selector == "date":
        return head == "DATE"
    if selector == "decimal":
        return head in {"DECIMAL", "NUMERIC"}
    if selector == "numeric":
        return head in _NUMERIC_HEAD_TYPES
    if selector == "boolean":
        return head == "BOOLEAN"
    if selector == "string":
        return head in _STRING_HEAD_TYPES
    return False


def scope_matches(scope: str, table_fqn: str) -> bool:
    """Whether *table_fqn* matches a policy ``scope`` (case-insensitive).

    Two flavours, chosen automatically:

    * **Glob** — when the scope contains a ``*``, it is the only wildcard and
      matches any run of characters (including dots). Every other character is
      matched literally. E.g. ``main.finance.*`` matches ``main.finance.orders``.
    * **Prefix** — when the scope has NO wildcard, it is treated as a
      dot-bounded prefix so an admin can scope a whole catalog or schema by name
      without remembering to append ``.*``. E.g. ``samples`` matches
      ``samples.bakehouse.reviews`` and ``main.finance`` matches
      ``main.finance.orders``, while ``main.fin`` does NOT match
      ``main.finance`` (the boundary is a dot, not a substring).

    A blank scope matches nothing (a policy with no scope is inert rather than
    global-by-accident).
    """
    pattern = (scope or "").strip()
    if not pattern:
        return False
    target = (table_fqn or "").strip()
    if "*" in pattern:
        regex = "^" + "".join(".*" if ch == "*" else re.escape(ch) for ch in pattern) + "$"
        return re.match(regex, target, flags=re.IGNORECASE) is not None
    # No wildcard: exact match, or a dot-bounded prefix (catalog/schema scope).
    lowered_pattern = pattern.lower()
    lowered_target = target.lower()
    return lowered_target == lowered_pattern or lowered_target.startswith(lowered_pattern + ".")


def normalize_enforcement(raw: object) -> CoverageEnforcement:
    """Coerce a stored/incoming enforcement value; unknown -> ``advisory``."""
    candidate = str(raw or "").strip().lower()
    if candidate == "blocking":
        return "blocking"
    return "advisory"


def normalize_column_type(raw: object) -> CoverageColumnType:
    """Coerce a stored/incoming column-type value; unknown -> ``any``."""
    candidate = str(raw or "").strip().lower()
    if candidate in COVERAGE_COLUMN_TYPES:
        return candidate  # type: ignore[return-value]
    return "any"


def evaluate_policy(
    policy: CoveragePolicy,
    columns: list[tuple[str, str, bool]],
    covered: set[tuple[str, str]],
) -> CoveragePolicyEvaluation:
    """Evaluate one policy against a table's columns and applied checks.

    Args:
        policy: the policy to evaluate.
        columns: ``(name, type_name, is_primary_key)`` for every table column.
        covered: set of ``(check_function_or_rule_id, column_name_lower)`` pairs
            currently applied to the table.

    Returns:
        A :class:`CoveragePolicyEvaluation`. A matched column is *covered* only
        when it carries EVERY required check; it is *missing* when at least one
        required check is absent. A policy with no matched columns (or no
        required checks) is trivially satisfied (nothing is missing).
    """
    matched: list[str] = []
    for name, type_name, is_pk in columns:
        if policy.primary_key_only and not is_pk:
            continue
        if column_type_matches(policy.column_type, type_name):
            matched.append(name)

    required_list = policy.effective_checks()
    covered_cols: list[str] = []
    missing_cols: list[str] = []
    for name in matched:
        if all((check, name.lower()) in covered for check in required_list):
            covered_cols.append(name)
        else:
            missing_cols.append(name)

    return CoveragePolicyEvaluation(
        policy_id=policy.policy_id,
        name=policy.name,
        enforcement=policy.enforcement,
        required_check=required_list[0] if required_list else "",
        required_checks=required_list,
        column_type=policy.column_type,
        primary_key_only=policy.primary_key_only,
        matched_columns=matched,
        covered_columns=covered_cols,
        missing_columns=missing_cols,
        satisfied=not missing_cols,
    )
