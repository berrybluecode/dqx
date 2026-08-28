"""Coverage-policy admin + compliance endpoints.

Coverage policies are rule-level, column-type-driven requirements ("every
``timestamp`` column on ``main.finance.*`` MUST have an ``is_not_null`` rule")
with an enforcement level of ``advisory`` or ``blocking``. Reads are open to any
authenticated user (the table page + compliance report surface them); writes
(policy CRUD + waivers) are admin-only, matching the other governance settings.

The blocking gate itself lives in the monitored-table approve/submit routes —
they call :meth:`CoveragePolicyService.enforce_blocking` before publishing. This
router is the read/write surface plus the per-binding compliance report.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from databricks_labs_dqx_app.backend.common.authorization import UserRole, get_user_email
from databricks_labs_dqx_app.backend.coverage_models import CoveragePolicy, CoverageReport, CoverageWaiver
from databricks_labs_dqx_app.backend.dependencies import (
    get_coverage_policy_service,
    get_monitored_table_service,
    require_role,
)
from databricks_labs_dqx_app.backend.logger import logger
from databricks_labs_dqx_app.backend.services.coverage_policy_service import CoveragePolicyService
from databricks_labs_dqx_app.backend.services.monitored_table_service import MonitoredTableService

router = APIRouter()


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class CoveragePoliciesOut(BaseModel):
    policies: list[CoveragePolicy]


class CoveragePoliciesIn(BaseModel):
    policies: list[CoveragePolicy]


class CoverageWaiversOut(BaseModel):
    waivers: list[CoverageWaiver]


class GrantWaiverIn(BaseModel):
    binding_id: str
    policy_id: str
    reason: str = ""


def _resolve_table_fqn(monitored_tables: MonitoredTableService, binding_id: str) -> str:
    detail = monitored_tables.get(binding_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Monitored table not found: {binding_id}")
    return detail.table.table_fqn


# ---------------------------------------------------------------------------
# Policy CRUD
# ---------------------------------------------------------------------------


@router.get(
    "/policies",
    response_model=CoveragePoliciesOut,
    operation_id="getCoveragePolicies",
)
def get_coverage_policies(
    svc: Annotated[CoveragePolicyService, Depends(get_coverage_policy_service)],
) -> CoveragePoliciesOut:
    """Return all coverage policies.

    Available to any authenticated user — the table page + compliance report
    need the policy list to show which coverage is expected. Only admins can
    change it (the PUT below).
    """
    return CoveragePoliciesOut(policies=svc.list_policies())


@router.put(
    "/policies",
    response_model=CoveragePoliciesOut,
    operation_id="saveCoveragePolicies",
    dependencies=[require_role(UserRole.ADMIN)],
)
def save_coverage_policies(
    body: CoveragePoliciesIn,
    svc: Annotated[CoveragePolicyService, Depends(get_coverage_policy_service)],
    email: Annotated[str, Depends(get_user_email)],
) -> CoveragePoliciesOut:
    """Replace the full set of coverage policies (admin only).

    New policies (blank ``policy_id``) get a server-minted id. Dropping a policy
    also prunes any waivers that referenced it. 400 on a validation failure
    (blank name/scope/required check, or an invalid enforcement/column type).
    """
    try:
        saved = svc.replace_policies(body.policies, user_email=email)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info("Saved %d coverage policy(ies) (by=%s)", len(saved), email)
    return CoveragePoliciesOut(policies=saved)


# ---------------------------------------------------------------------------
# Per-binding compliance report
# ---------------------------------------------------------------------------


@router.get(
    "/report/{binding_id}",
    response_model=CoverageReport,
    operation_id="getCoverageReport",
)
def get_coverage_report(
    binding_id: str,
    svc: Annotated[CoveragePolicyService, Depends(get_coverage_policy_service)],
    monitored_tables: Annotated[MonitoredTableService, Depends(get_monitored_table_service)],
) -> CoverageReport:
    """Evaluate every enabled, in-scope policy against the binding's table.

    Read-only; available to any authenticated user. Surfaces missing coverage
    (advisory + blocking alike) and flags which blocking policies would refuse
    an approval.
    """
    table_fqn = _resolve_table_fqn(monitored_tables, binding_id)
    return svc.build_report(binding_id, table_fqn)


# ---------------------------------------------------------------------------
# Waivers
# ---------------------------------------------------------------------------


@router.get(
    "/waivers/{binding_id}",
    response_model=CoverageWaiversOut,
    operation_id="getCoverageWaivers",
)
def get_coverage_waivers(
    binding_id: str,
    svc: Annotated[CoveragePolicyService, Depends(get_coverage_policy_service)],
) -> CoverageWaiversOut:
    """Return the coverage waivers granted for a binding (any authenticated user)."""
    return CoverageWaiversOut(waivers=svc.list_waivers(binding_id))


@router.post(
    "/waivers",
    response_model=CoverageWaiver,
    operation_id="grantCoverageWaiver",
    dependencies=[require_role(UserRole.ADMIN)],
)
def grant_coverage_waiver(
    body: GrantWaiverIn,
    svc: Annotated[CoveragePolicyService, Depends(get_coverage_policy_service)],
    email: Annotated[str, Depends(get_user_email)],
) -> CoverageWaiver:
    """Grant (or update) a waiver of one policy for one binding (admin only).

    A waiver suppresses that policy's blocking enforcement for the binding, so
    the binding can be approved despite unmet coverage. 404 when the policy id
    is unknown.
    """
    if svc.get_policy(body.policy_id) is None:
        raise HTTPException(status_code=404, detail=f"Coverage policy not found: {body.policy_id}")
    waiver = svc.grant_waiver(body.binding_id, body.policy_id, body.reason, user_email=email)
    logger.info("Granted coverage waiver (binding=%s policy=%s by=%s)", body.binding_id, body.policy_id, email)
    return waiver


@router.delete(
    "/waivers/{binding_id}/{policy_id}",
    operation_id="revokeCoverageWaiver",
    dependencies=[require_role(UserRole.ADMIN)],
)
def revoke_coverage_waiver(
    binding_id: str,
    policy_id: str,
    svc: Annotated[CoveragePolicyService, Depends(get_coverage_policy_service)],
    email: Annotated[str, Depends(get_user_email)],
) -> dict[str, bool]:
    """Revoke a waiver (admin only). Returns ``{"revoked": bool}``."""
    revoked = svc.revoke_waiver(binding_id, policy_id, user_email=email)
    logger.info("Revoked coverage waiver (binding=%s policy=%s by=%s)", binding_id, policy_id, email)
    return {"revoked": revoked}
