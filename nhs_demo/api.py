from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from nhs_demo.orchestrator import DemoOrchestrator
from nhs_demo.schemas import (
    FormIntakeRequest,
    IntakeRequest,
    IntakeRefineRequest,
    IntakeResponse,
    RouteRequest,
    RouteResponse,
    ScheduleOfferRequest,
    ScheduleOfferResponse,
    ScheduleRelaxRequest,
    ScheduleRelaxResponse,
    SlotInventoryResponse,
)


def _load_local_env() -> None:
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_local_env()


def create_app() -> FastAPI:
    app = FastAPI(title="NHS-Style Multi-Agent Scheduling Demo", version="1.0.0")
    orchestrator = DemoOrchestrator()

    package_root = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(package_root / "templates"))
    app.mount("/static", StaticFiles(directory=str(package_root / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/audit/{run_id}", response_class=HTMLResponse)
    def audit_page(run_id: str, request: Request) -> HTMLResponse:
        try:
            audit = orchestrator.audit(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return templates.TemplateResponse(
            "audit.html",
            {
                "request": request,
                "audit": audit.model_dump(mode="json"),
                "run_id": run_id,
            },
        )

    @app.post("/api/intake", response_model=IntakeResponse)
    def intake(payload: IntakeRequest) -> IntakeResponse:
        try:
            return orchestrator.intake(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/intake/refine", response_model=IntakeResponse)
    def intake_refine(payload: IntakeRefineRequest) -> IntakeResponse:
        try:
            return orchestrator.refine_intake(payload)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/intake/form", response_model=IntakeResponse)
    def intake_form(payload: FormIntakeRequest) -> IntakeResponse:
        try:
            return orchestrator.intake_form(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/route", response_model=RouteResponse)
    def route(payload: RouteRequest) -> RouteResponse:
        try:
            return orchestrator.route(payload)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/schedule/offer", response_model=ScheduleOfferResponse)
    def schedule_offer(payload: ScheduleOfferRequest) -> ScheduleOfferResponse:
        try:
            return orchestrator.offer(payload)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/schedule/preview", response_model=ScheduleOfferResponse)
    def schedule_preview(payload: ScheduleOfferRequest) -> ScheduleOfferResponse:
        try:
            return orchestrator.preview_offer(payload)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/schedule/relax", response_model=ScheduleRelaxResponse)
    def schedule_relax(payload: ScheduleRelaxRequest) -> ScheduleRelaxResponse:
        try:
            return orchestrator.relax(payload)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/run/{run_id}/audit")
    def get_audit(run_id: str):
        try:
            return orchestrator.audit(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/slots", response_model=SlotInventoryResponse)
    def get_slot_inventory(
        service_type: str | None = None,
        horizon_days: int | None = None,
    ) -> SlotInventoryResponse:
        try:
            return orchestrator.slot_inventory(
                service_type=service_type,
                horizon_days=horizon_days,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
