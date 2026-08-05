"""FastAPI application factory for the asynchronous AudioDiT service."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app_config import (
    DEFAULT_API_DATA_DIR,
    DEFAULT_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_MAX_PENDING_TASKS,
    DEFAULT_RESULT_TTL_HOURS,
    MAX_REFERENCE_UPLOAD_BYTES,
    MAX_REFERENCE_TEXT_CHARS,
    REFERENCE_UPLOAD_CHUNK_BYTES,
)
from .schemas import GenerateAudioRequest
from .task_manager import (
    AudioValidationError,
    PersistentTaskManager,
    QueueLimitError,
    ReferenceConflictError,
    ServiceNotReadyError,
    UnsafePathError,
)
from .validation import validate_resource_id


def create_app(
    service,
    data_dir: str | Path = DEFAULT_API_DATA_DIR,
    result_ttl_hours: float = DEFAULT_RESULT_TTL_HOURS,
    cleanup_interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
    max_pending_tasks: int = DEFAULT_MAX_PENDING_TASKS,
) -> FastAPI:
    manager = PersistentTaskManager(
        service=service,
        data_dir=data_dir,
        result_ttl_hours=result_ttl_hours,
        cleanup_interval_seconds=cleanup_interval_seconds,
        max_pending_tasks=max_pending_tasks,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(title="LongCat AudioDiT API", version="1.0.0", lifespan=lifespan)
    app.state.task_manager = manager

    @app.post("/v1/references/add")
    async def add_reference(
        id: str = Form(...),
        audio: UploadFile = File(...),
        text: str = Form(...),
    ):
        try:
            reference_id = validate_resource_id(id, "reference_id")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        transcript = (text or "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="reference transcript cannot be empty")
        if len(transcript) > MAX_REFERENCE_TEXT_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"reference transcript exceeds {MAX_REFERENCE_TEXT_CHARS} characters",
            )

        handle, staging_name = tempfile.mkstemp(
            prefix="reference-upload-", dir=manager.staging_dir
        )
        staging = Path(staging_name)
        digest = hashlib.sha256()
        total = 0
        try:
            with os.fdopen(handle, "wb") as output:
                while chunk := await audio.read(REFERENCE_UPLOAD_CHUNK_BYTES):
                    total += len(chunk)
                    if total > MAX_REFERENCE_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "reference audio exceeds "
                                f"{MAX_REFERENCE_UPLOAD_BYTES // (1024 * 1024)} MiB"
                            ),
                        )
                    output.write(chunk)
                    digest.update(chunk)
            if total == 0:
                raise HTTPException(
                    status_code=422, detail="reference audio cannot be empty"
                )
            digest.update(b"\0")
            digest.update(transcript.encode("utf-8"))
            reference = await run_in_threadpool(
                manager.add_reference_from_file,
                reference_id,
                staging,
                transcript,
                digest.hexdigest(),
            )
        except HTTPException:
            raise
        except ReferenceConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except AudioValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            staging.unlink(missing_ok=True)
            await audio.close()
        return {
            "success": True,
            "reference_id": reference["id"],
            "content_sha256": reference["content_sha256"],
            "reused": reference["reused"],
        }

    @app.post("/generate_audio_enhanced_async")
    async def generate_audio(request: GenerateAudioRequest):
        try:
            task, created = await run_in_threadpool(manager.submit, request.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=error.args[0]) from error
        except QueueLimitError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except ServiceNotReadyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        response = manager.status_dict(task)
        if created:
            # The acceptance contract is stable even if the FIFO worker starts
            # the task before this HTTP response is serialized.
            response["status"] = "pending"
            response["started_at"] = None
            response["completed_at"] = None
            response["download_url"] = None
            response["error"] = None
        response["created"] = created
        return response

    @app.get("/get_task_status")
    async def get_task_status(step_id: str):
        try:
            task = manager.get(step_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return manager.status_dict(task)

    @app.get("/download_result")
    async def download_result(step_id: str):
        try:
            task = manager.get(step_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        if task.status != "completed":
            raise HTTPException(
                status_code=409, detail=f"task is not completed: {task.status}"
            )
        try:
            result = manager.result_file(task)
        except (FileNotFoundError, UnsafePathError):
            raise HTTPException(status_code=410, detail="result file is no longer available")
        return FileResponse(
            result,
            media_type="audio/mpeg",
            filename=f"{task.step_id}.mp3",
        )

    @app.post("/stop_async_task/{step_id}")
    async def stop_task(step_id: str):
        try:
            task = manager.cancel(step_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return manager.status_dict(task)

    @app.get("/v1/health")
    async def health():
        status = manager.health_status()
        return JSONResponse(status, status_code=200 if status["status"] == "ok" else 503)

    return app


__all__ = ["create_app"]
