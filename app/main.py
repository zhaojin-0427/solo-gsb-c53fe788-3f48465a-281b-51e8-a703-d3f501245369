"""FastAPI application for chromatographic quantification."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .quant import solve

app = FastAPI(
    title="Chromatographic Quantification API",
    version="1.0.0",
    description=(
        "Time-warp anchored, interval-safe quantitative analysis of "
        "chromatographic time/intensity curves (up to six pure standards)."
    ),
)


@app.get("/")
def root() -> dict:
    return {
        "service": "chromatography-quant",
        "docs": "/docs",
        "endpoint": {"path": "/quantify", "method": "POST"},
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/quantify")
def quantify(payload: dict) -> JSONResponse:
    result = solve(payload)
    # Validation errors -> 400; analytical statuses (ok, insufficient_anchors,
    # infeasible, exhaustion) are well-formed answers -> 200.
    status_code = 400 if result.get("status") == "invalid" else 200
    return JSONResponse(result, status_code=status_code)
