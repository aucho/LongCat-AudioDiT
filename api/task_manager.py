"""Persistent single-worker FIFO task management for AudioDiT generation."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app_config import (
    DEFAULT_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_MAX_PENDING_TASKS,
    DEFAULT_MP3_BITRATE,
    DEFAULT_RESULT_TTL_HOURS,
    DEFAULT_SPEECH_RATE,
    MAX_GENERATION_TEXT_CHARS,
    MAX_REFERENCE_DURATION_SECONDS,
    MAX_REFERENCE_TEXT_CHARS,
)
from services import GenerationCancelledError
from utils import normalize_text, normalize_tts_text

from .validation import validate_resource_id


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"pending", "processing"}


class ReferenceConflictError(ValueError):
    """Raised when an existing reference id is reused with different content."""


class AudioValidationError(ValueError):
    """Raised when an uploaded file cannot be accepted as reference audio."""


class QueueLimitError(RuntimeError):
    """Raised when the configured pending-task admission limit is reached."""


class ServiceNotReadyError(RuntimeError):
    """Raised when durable task processing is no longer healthy."""


class UnsafePathError(ValueError):
    """Raised when persisted state points outside its controlled directory."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _chmod_private(path: Path, directory: bool = False) -> None:
    if os.name == "posix":
        path.chmod(0o700 if directory else 0o600)


@dataclass
class TaskRecord:
    step_id: str
    request: dict[str, Any]
    status: str = "pending"
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result_path: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "request": self.request,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "error": self.error,
            "result_path": self.result_path,
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskRecord":
        return cls(
            step_id=value["step_id"],
            request=value["request"],
            status=value.get("status", "failed"),
            created_at=_parse_time(value.get("created_at")) or _utc_now(),
            started_at=_parse_time(value.get("started_at")),
            completed_at=_parse_time(value.get("completed_at")),
            error=value.get("error"),
            result_path=value.get("result_path"),
            progress_current=int(value.get("progress_current", 0)),
            progress_total=int(value.get("progress_total", 0)),
        )


class PersistentTaskManager:
    """Persist task state and execute generation with exactly one worker thread."""

    def __init__(
        self,
        service,
        data_dir: str | Path,
        result_ttl_hours: float = DEFAULT_RESULT_TTL_HOURS,
        cleanup_interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
        max_pending_tasks: int = DEFAULT_MAX_PENDING_TASKS,
        start_worker: bool = True,
    ):
        if result_ttl_hours <= 0:
            raise ValueError("result_ttl_hours must be greater than 0")
        if max_pending_tasks <= 0:
            raise ValueError("max_pending_tasks must be greater than 0")
        self.service = service
        self.data_dir = Path(data_dir).resolve()
        self.tasks_dir = self.data_dir / "tasks"
        self.results_dir = self.data_dir / "results"
        self.references_dir = self.data_dir / "references"
        self.staging_dir = self.data_dir / "staging"
        for directory in (
            self.data_dir,
            self.tasks_dir,
            self.results_dir,
            self.references_dir,
            self.staging_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            _chmod_private(directory, directory=True)
        self._verify_data_directory()
        self.result_ttl = timedelta(hours=float(result_ttl_hours))
        self.cleanup_interval_seconds = max(float(cleanup_interval_seconds), 0.1)
        self.max_pending_tasks = int(max_pending_tasks)
        self._tasks: dict[str, TaskRecord] = {}
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._current_step_id: str | None = None
        self._fatal_error: str | None = None
        self._last_worker_error: str | None = None
        self._last_cleanup_error: str | None = None
        self._worker: threading.Thread | None = None
        self._cleaner: threading.Thread | None = None
        self._load_tasks()
        self.cleanup_expired()
        if start_worker:
            self.start()

    def _verify_data_directory(self) -> None:
        handle, name = tempfile.mkstemp(prefix=".write-test-", dir=self.staging_dir)
        os.close(handle)
        probe = Path(name)
        try:
            replacement = probe.with_suffix(".ok")
            os.replace(probe, replacement)
            replacement.unlink()
        finally:
            probe.unlink(missing_ok=True)

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name="longcat-fifo-worker", daemon=True
        )
        self._cleaner = threading.Thread(
            target=self._cleanup_loop, name="longcat-result-cleaner", daemon=True
        )
        self._worker.start()
        self._cleaner.start()

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._worker:
            self._worker.join(timeout=timeout)
        if self._cleaner:
            self._cleaner.join(timeout=timeout)

    @property
    def current_step_id(self) -> str | None:
        with self._lock:
            return self._current_step_id

    @property
    def queue_size(self) -> int:
        with self._lock:
            return sum(task.status == "pending" for task in self._tasks.values())

    @property
    def processing_count(self) -> int:
        with self._lock:
            return sum(task.status == "processing" for task in self._tasks.values())

    def _manifest_path(self, step_id: str) -> Path:
        return self.tasks_dir / f"{validate_resource_id(step_id, 'step_id')}.json"

    def _reference_dir(self, reference_id: str) -> Path:
        return self.references_dir / validate_resource_id(reference_id, "reference_id")

    def _relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.data_dir).as_posix()

    def _resolve_controlled_path(
        self,
        stored_path: str,
        root: Path,
        *,
        exact: Path | None = None,
    ) -> Path:
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = self.data_dir / candidate
        resolved = candidate.resolve()
        controlled_root = root.resolve()
        try:
            resolved.relative_to(controlled_root)
        except ValueError as error:
            raise UnsafePathError(f"path is outside {controlled_root}") from error
        if exact is not None and resolved != exact.resolve():
            raise UnsafePathError(f"path does not match expected file {exact}")
        return resolved

    def _expected_result_path(self, step_id: str) -> Path:
        return (self.results_dir / f"{validate_resource_id(step_id, 'step_id')}.mp3").resolve()

    def result_file(self, task: TaskRecord) -> Path:
        if not task.result_path:
            raise UnsafePathError("task has no result path")
        result = self._resolve_controlled_path(
            task.result_path,
            self.results_dir,
            exact=self._expected_result_path(task.step_id),
        )
        if not result.is_file():
            raise FileNotFoundError(result)
        return result

    def _persist(self, task: TaskRecord) -> None:
        path = self._manifest_path(task.step_id)
        temporary = path.with_suffix(".json.tmp")
        try:
            if task.result_path:
                controlled_result = self._resolve_controlled_path(
                    task.result_path,
                    self.results_dir,
                    exact=self._expected_result_path(task.step_id),
                )
                task.result_path = self._relative_path(controlled_result)
            temporary.write_text(
                json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _chmod_private(temporary)
            os.replace(temporary, path)
        except Exception as error:
            self._fatal_error = f"task manifest persistence failed: {error}"
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def _load_tasks(self) -> None:
        for manifest in self.tasks_dir.glob("*.json"):
            try:
                task = TaskRecord.from_dict(
                    json.loads(manifest.read_text(encoding="utf-8"))
                )
                validate_resource_id(task.step_id, "step_id")
                if manifest.resolve() != self._manifest_path(task.step_id).resolve():
                    raise UnsafePathError("manifest filename does not match step_id")
                if task.status in ACTIVE_STATUSES:
                    task.status = "failed"
                    task.error = "service restarted before task completion"
                    task.completed_at = _utc_now()
                    task.result_path = None
                    self._persist(task)
                elif task.status == "completed":
                    try:
                        result = self.result_file(task)
                        task.result_path = self._relative_path(result)
                    except (FileNotFoundError, UnsafePathError):
                        task.status = "failed"
                        task.error = "completed result path is missing or unsafe"
                        task.completed_at = _utc_now()
                        task.result_path = None
                    self._persist(task)
                elif task.result_path:
                    try:
                        result = self._resolve_controlled_path(
                            task.result_path,
                            self.results_dir,
                            exact=self._expected_result_path(task.step_id),
                        )
                        task.result_path = self._relative_path(result)
                    except UnsafePathError:
                        task.result_path = None
                    self._persist(task)
                self._tasks[task.step_id] = task
            except Exception:
                # Corrupt manifests remain on disk for diagnosis and are never queued.
                continue

    @staticmethod
    def _reference_hash(audio_path: Path, text: str) -> str:
        digest = hashlib.sha256()
        with audio_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        digest.update(text.strip().encode("utf-8"))
        return digest.hexdigest()

    def _probe_reference_audio(self, source: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type,duration:format=duration",
            "-of",
            "json",
            str(source),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True, timeout=30
            )
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            raise AudioValidationError(f"reference audio is not readable: {error}") from error
        streams = payload.get("streams") or []
        if not streams:
            raise AudioValidationError("reference file contains no readable audio stream")
        raw_duration = (payload.get("format") or {}).get("duration") or streams[0].get(
            "duration"
        )
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError) as error:
            raise AudioValidationError("reference audio duration is unavailable") from error
        if duration <= 0:
            raise AudioValidationError("reference audio duration must be greater than zero")
        if duration > MAX_REFERENCE_DURATION_SECONDS:
            raise AudioValidationError(
                f"reference audio must not exceed {MAX_REFERENCE_DURATION_SECONDS:g} seconds"
            )
        return duration

    def _normalize_reference_audio(self, source: Path, destination: Path) -> None:
        temporary = destination.with_name("audio.tmp.wav")
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=120)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise AudioValidationError("FFmpeg produced an empty reference WAV")
            _chmod_private(temporary)
            os.replace(temporary, destination)
        except OSError as error:
            raise AudioValidationError(f"FFmpeg is unavailable: {error}") from error
        except subprocess.SubprocessError as error:
            raise AudioValidationError("reference audio could not be decoded") from error
        finally:
            temporary.unlink(missing_ok=True)

    def add_reference_from_file(
        self,
        reference_id: str,
        staging_path: str | Path,
        text: str,
        content_sha256: str,
    ) -> dict[str, Any]:
        reference_id = validate_resource_id(reference_id, "reference_id")
        transcript = (text or "").strip()
        if not transcript:
            raise ValueError("reference transcript cannot be empty")
        if len(transcript) > MAX_REFERENCE_TEXT_CHARS:
            raise ValueError(
                f"reference transcript exceeds {MAX_REFERENCE_TEXT_CHARS} characters"
            )
        if len(content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in content_sha256.lower()
        ):
            raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
        content_sha256 = content_sha256.lower()
        source = Path(staging_path)
        reference_dir = self._reference_dir(reference_id)
        metadata_path = reference_dir / "metadata.json"
        try:
            with self._lock:
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    existing = self._validated_reference(metadata, reference_id)
                    existing_hash = metadata.get("content_sha256")
                    if not existing_hash:
                        existing_hash = self._reference_hash(
                            Path(existing["audio_path"]), metadata.get("text", "")
                        )
                        metadata["content_sha256"] = existing_hash
                        metadata["audio_path"] = self._relative_path(
                            Path(existing["audio_path"])
                        )
                        self._write_reference_metadata(metadata_path, metadata)
                    if existing_hash != content_sha256:
                        raise ReferenceConflictError(
                            "reference_id already exists with different audio or transcript"
                        )
                    existing["reused"] = True
                    return existing

                duration = self._probe_reference_audio(source)
                reference_dir.mkdir(parents=True, exist_ok=True)
                _chmod_private(reference_dir, directory=True)
                audio_path = reference_dir / "audio.wav"
                self._normalize_reference_audio(source, audio_path)
                metadata = {
                    "id": reference_id,
                    "audio_path": self._relative_path(audio_path),
                    "text": transcript,
                    "content_sha256": content_sha256,
                    "duration_seconds": duration,
                    "created_at": _iso(_utc_now()),
                }
                self._write_reference_metadata(metadata_path, metadata)
                result = dict(metadata)
                result["audio_path"] = str(audio_path.resolve())
                result["reused"] = False
                return result
        finally:
            source.unlink(missing_ok=True)

    def _write_reference_metadata(
        self, metadata_path: Path, metadata: dict[str, Any]
    ) -> None:
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _chmod_private(temporary)
        os.replace(temporary, metadata_path)

    def add_reference(
        self, reference_id: str, audio: bytes, filename: str, text: str
    ) -> dict[str, Any]:
        del filename
        handle, name = tempfile.mkstemp(prefix="reference-", dir=self.staging_dir)
        staging = Path(name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(audio)
            digest = hashlib.sha256(audio + b"\0" + (text or "").strip().encode("utf-8"))
            return self.add_reference_from_file(
                reference_id, staging, text, digest.hexdigest()
            )
        finally:
            staging.unlink(missing_ok=True)

    def _validated_reference(
        self, metadata: dict[str, Any], reference_id: str
    ) -> dict[str, Any]:
        if metadata.get("id") != reference_id:
            raise UnsafePathError("reference metadata id does not match directory")
        transcript = metadata.get("text")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("reference transcript is missing")
        if len(transcript) > MAX_REFERENCE_TEXT_CHARS:
            raise ValueError("reference transcript is too large")
        reference_dir = self._reference_dir(reference_id).resolve()
        audio_path = self._resolve_controlled_path(
            str(metadata.get("audio_path", "")), reference_dir
        )
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        result = dict(metadata)
        result["audio_path"] = str(audio_path)
        return result

    def get_reference(self, reference_id: str) -> dict[str, Any] | None:
        reference_id = validate_resource_id(reference_id, "reference_id")
        metadata_path = self._reference_dir(reference_id) / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return self._validated_reference(metadata, reference_id)
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            return None

    def _validate_generation_text(self, request: dict[str, Any]) -> None:
        text = str(request.get("text", ""))
        if not text.strip():
            raise ValueError("text cannot be empty")
        if len(text) > MAX_GENERATION_TEXT_CHARS:
            raise ValueError(f"text exceeds {MAX_GENERATION_TEXT_CHARS} characters")
        language = str(request.get("language", ""))
        if hasattr(self.service, "normalize_spoken_text"):
            spoken = self.service.normalize_spoken_text(text, language, "text")
        else:
            spoken = normalize_text(normalize_tts_text(text, language).spoken_text)
        if len(spoken) > MAX_GENERATION_TEXT_CHARS:
            raise ValueError(
                f"normalized text exceeds {MAX_GENERATION_TEXT_CHARS} characters"
            )

    def submit(self, request: dict[str, Any]) -> tuple[TaskRecord, bool]:
        step_id = validate_resource_id(request.get("step_id", ""), "step_id")
        with self._lock:
            existing = self._tasks.get(step_id)
            if existing and existing.status in ACTIVE_STATUSES | {"completed"}:
                return existing, False
            if self._fatal_error:
                raise ServiceNotReadyError(self._fatal_error)
            if not bool(getattr(self.service, "is_ready", False)):
                raise ServiceNotReadyError(
                    str(getattr(self.service, "load_error", "model is not loaded"))
                )
            if self._worker is not None and not self._worker.is_alive():
                raise ServiceNotReadyError("task worker is not running")

        reference_id = validate_resource_id(
            request.get("reference_id", ""), "reference_id"
        )
        if not self.get_reference(reference_id):
            raise KeyError(f"reference not found: {reference_id}")
        self._validate_generation_text(request)

        with self._lock:
            existing = self._tasks.get(step_id)
            if existing and existing.status in ACTIVE_STATUSES | {"completed"}:
                return existing, False
            if self.queue_size >= self.max_pending_tasks:
                raise QueueLimitError(
                    f"pending task limit reached ({self.max_pending_tasks})"
                )
            if existing:
                if existing.result_path:
                    try:
                        self._resolve_controlled_path(
                            existing.result_path,
                            self.results_dir,
                            exact=self._expected_result_path(step_id),
                        ).unlink(missing_ok=True)
                    except UnsafePathError:
                        pass
                self._manifest_path(step_id).unlink(missing_ok=True)
            task = TaskRecord(step_id=step_id, request=dict(request))
            self._tasks[step_id] = task
            try:
                self._persist(task)
            except Exception:
                del self._tasks[step_id]
                raise
            self._queue.put((step_id, task.created_at.isoformat()))
            return task, True

    def get(self, step_id: str) -> TaskRecord | None:
        step_id = validate_resource_id(step_id, "step_id")
        with self._lock:
            return self._tasks.get(step_id)

    def cancel(self, step_id: str) -> TaskRecord | None:
        step_id = validate_resource_id(step_id, "step_id")
        with self._lock:
            task = self._tasks.get(step_id)
            if not task:
                return None
            if task.status == "pending":
                task.cancel_event.set()
                task.status = "cancelled"
                task.completed_at = _utc_now()
                task.error = "cancelled by user"
                self._persist(task)
            elif task.status == "processing":
                task.cancel_event.set()
            return task

    def status_dict(self, task: TaskRecord) -> dict[str, Any]:
        with self._lock:
            return {
                "step_id": task.step_id,
                "status": task.status,
                "created_at": _iso(task.created_at),
                "started_at": _iso(task.started_at),
                "completed_at": _iso(task.completed_at),
                "progress": {
                    "current": task.progress_current,
                    "total": task.progress_total,
                },
                "download_url": (
                    f"/download_result?step_id={task.step_id}"
                    if task.status == "completed"
                    else None
                ),
                "error": task.error,
                "cancel_requested": task.cancel_event.is_set(),
            }

    def cleanup_expired(self, now: datetime | None = None) -> list[str]:
        now = now or _utc_now()
        removed: list[str] = []
        with self._lock:
            for step_id, task in list(self._tasks.items()):
                if task.status not in TERMINAL_STATUSES or not task.completed_at:
                    continue
                if now - task.completed_at < self.result_ttl:
                    continue
                if task.result_path:
                    try:
                        self._resolve_controlled_path(
                            task.result_path,
                            self.results_dir,
                            exact=self._expected_result_path(step_id),
                        ).unlink(missing_ok=True)
                    except UnsafePathError:
                        pass
                self._manifest_path(step_id).unlink(missing_ok=True)
                del self._tasks[step_id]
                removed.append(step_id)
        return removed

    def _update_progress(self, task: TaskRecord, current: int, total: int) -> None:
        with self._lock:
            task.progress_current = int(current)
            task.progress_total = int(total)
            self._persist(task)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            queue_item = self._queue.get()
            if queue_item is None:
                self._queue.task_done()
                break
            try:
                self._run_task(*queue_item)
            except Exception as error:
                with self._lock:
                    self._last_worker_error = str(error)
                    self._fatal_error = f"worker stopped after persistence failure: {error}"
                    self._current_step_id = None
                break
            finally:
                self._queue.task_done()

    def _run_task(self, step_id: str, task_token: str) -> None:
        with self._lock:
            task = self._tasks.get(step_id)
            if (
                not task
                or task.status != "pending"
                or task.created_at.isoformat() != task_token
            ):
                return
            task.status = "processing"
            task.started_at = _utc_now()
            self._current_step_id = step_id
            self._persist(task)
        generated_path: Path | None = None
        generated_dir: Path | None = None
        try:
            reference = self.get_reference(task.request["reference_id"])
            if not reference:
                raise FileNotFoundError("reference audio is missing or unsafe")
            arguments = dict(task.request)
            arguments.pop("step_id", None)
            arguments.pop("reference_id", None)
            if arguments.get("speech_rate") is None:
                arguments["speech_rate"] = DEFAULT_SPEECH_RATE
            generated = self.service.generate_long_audio(
                prompt_audio_path=reference["audio_path"],
                prompt_text=reference["text"],
                bitrate=DEFAULT_MP3_BITRATE,
                should_cancel=task.cancel_event.is_set,
                progress_callback=lambda current, total: self._update_progress(
                    task, current, total
                ),
                **arguments,
            )
            generated_path = Path(generated[1]).resolve()
            generated_dir = generated_path.parent
            if task.cancel_event.is_set():
                raise GenerationCancelledError("generation cancelled")
            result_path = self._expected_result_path(step_id)
            temporary = result_path.with_name(f".{step_id}.mp3.tmp")
            shutil.copyfile(generated_path, temporary)
            _chmod_private(temporary)
            os.replace(temporary, result_path)
            if task.cancel_event.is_set():
                result_path.unlink(missing_ok=True)
                raise GenerationCancelledError("generation cancelled")
            with self._lock:
                task.status = "completed"
                task.result_path = self._relative_path(result_path)
                task.completed_at = _utc_now()
                task.error = None
                self._persist(task)
        except GenerationCancelledError as error:
            with self._lock:
                task.status = "cancelled"
                task.completed_at = _utc_now()
                task.error = str(error)
                self._persist(task)
        except Exception as error:
            with self._lock:
                task.status = "failed"
                task.completed_at = _utc_now()
                task.error = str(error)
                self._persist(task)
        finally:
            if generated_dir and generated_dir.name.startswith("longcat_long_"):
                shutil.rmtree(generated_dir, ignore_errors=True)
            with self._lock:
                self._current_step_id = None

    def _cleanup_loop(self) -> None:
        while not self._stop.wait(self.cleanup_interval_seconds):
            try:
                self.cleanup_expired()
                with self._lock:
                    self._last_cleanup_error = None
            except Exception as error:
                with self._lock:
                    self._last_cleanup_error = str(error)

    def health_status(self) -> dict[str, Any]:
        with self._lock:
            model_loaded = bool(getattr(self.service, "is_ready", False))
            worker_alive = bool(self._worker and self._worker.is_alive())
            cleaner_alive = bool(self._cleaner and self._cleaner.is_alive())
            ready = model_loaded and worker_alive and cleaner_alive and not self._fatal_error
            device = getattr(self.service, "device", None)
            return {
                "status": "ok" if ready else "unavailable",
                "model_loaded": model_loaded,
                "model": getattr(self.service, "model_name", None),
                "device": str(device) if device is not None else None,
                "worker_alive": worker_alive,
                "cleaner_alive": cleaner_alive,
                "current_task": self._current_step_id,
                "queue_length": self.queue_size,
                "processing_count": self.processing_count,
                "last_worker_error": self._last_worker_error,
                "last_cleanup_error": self._last_cleanup_error,
                "fatal_error": self._fatal_error,
                "model_error": getattr(self.service, "load_error", None),
            }


__all__ = [
    "ACTIVE_STATUSES",
    "AudioValidationError",
    "PersistentTaskManager",
    "QueueLimitError",
    "ReferenceConflictError",
    "ServiceNotReadyError",
    "TERMINAL_STATUSES",
    "TaskRecord",
    "UnsafePathError",
    "validate_resource_id",
]
