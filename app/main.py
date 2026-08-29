from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import Settings, settings as default_settings
from app.database import Database
from app.features import ImageValidationError
from app.model_service import ModelNotReadyError, ModelService
from app.service import AnalysisService


def error_response(code: str, message: str, status_code: int, request_id: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
    )


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    active_settings = custom_settings or default_settings

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active_settings.ensure_directories()
        database = Database(active_settings.database_path)
        database.initialize()
        model = ModelService(active_settings.model_path)
        model_error = None
        try:
            model.load()
        except ModelNotReadyError as exc:
            model_error = str(exc)
        application.state.settings = active_settings
        application.state.database = database
        application.state.model = model
        application.state.model_error = model_error
        application.state.analysis_service = AnalysisService(active_settings, database, model)
        yield

    app = FastAPI(
        title="ImageGuard AI",
        description="Explainable image quality and visual anomaly assessment.",
        version=__version__,
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=active_settings.static_dir), name="static")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        import uuid

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return error_response("INVALID_REQUEST", "The request parameters are invalid.", 422, request.state.request_id)

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            return error_response(exc.detail["code"], exc.detail["message"], exc.status_code, request.state.request_id)
        return error_response("HTTP_ERROR", str(exc.detail), exc.status_code, request.state.request_id)

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        return error_response(
            "INTERNAL_ERROR",
            "The analysis could not be completed due to an unexpected server error.",
            500,
            getattr(request.state, "request_id", None),
        )

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(active_settings.static_dir / "index.html")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        database_ready = request.app.state.database.ready()
        model_ready = request.app.state.model.ready
        status_code = 200 if database_ready and model_ready else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ok" if status_code == 200 else "not_ready",
                "database": "ready" if database_ready else "unavailable",
                "model": "ready" if model_ready else "unavailable",
                "model_name": request.app.state.model.model_name,
                "model_version": request.app.state.model.model_version,
                "detail": request.app.state.model_error,
            },
        )

    @app.post("/api/v1/analyses", status_code=201)
    async def create_analysis(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        if not request.app.state.model.ready:
            raise HTTPException(503, {"code": "MODEL_NOT_READY", "message": request.app.state.model_error})
        max_bytes = active_settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        try:
            return request.app.state.analysis_service.analyze(
                data, file.filename or "upload", file.content_type
            )
        except ImageValidationError as exc:
            raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc

    @app.get("/api/v1/analyses")
    async def history(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return request.app.state.analysis_service.history(limit, offset)

    @app.get("/api/v1/analyses/{analysis_id}")
    async def analysis_detail(request: Request, analysis_id: str) -> dict[str, Any]:
        record = request.app.state.analysis_service.get(analysis_id)
        if not record:
            raise HTTPException(404, {"code": "ANALYSIS_NOT_FOUND", "message": "Analysis not found."})
        return record

    @app.get("/api/v1/analyses/{analysis_id}/image", include_in_schema=False)
    async def analysis_image(request: Request, analysis_id: str) -> FileResponse:
        record = request.app.state.analysis_service.get(analysis_id, public=False)
        if not record:
            raise HTTPException(404, {"code": "ANALYSIS_NOT_FOUND", "message": "Analysis not found."})
        image_path = active_settings.upload_dir / record["stored_filename"]
        if not image_path.is_file():
            raise HTTPException(404, {"code": "IMAGE_NOT_FOUND", "message": "Stored preview is unavailable."})
        return FileResponse(image_path, media_type="image/jpeg", filename=f"analysis-{analysis_id}.jpg")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=default_settings.host, port=default_settings.port, reload=True)

