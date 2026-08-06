import io
import json
import tempfile
import threading
import time
import unittest
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import PersistentTaskManager, create_app
from api.task_manager import TaskRecord
from services import GenerationCancelledError


def wav_bytes(sample=0, frames=800):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(int(sample).to_bytes(2, "little", signed=True) * frames)
    return output.getvalue()


def wait_for_status(client, step_id, expected, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/get_task_status", params={"step_id": step_id})
        if response.status_code == 200 and response.json()["status"] in expected:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"task {step_id} did not reach {expected}")


class FakeService:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.running = 0
        self.max_running = 0
        self.is_ready = True
        self.model_name = "fake-audiodit"
        self.device = "cpu"

    def generate_long_audio(self, **kwargs):
        self.calls.append(kwargs)
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.started.set()
        try:
            if self.block:
                while not self.release.wait(0.01):
                    if kwargs["should_cancel"]():
                        raise GenerationCancelledError("generation cancelled")
            kwargs["progress_callback"](1, 1)
            output_dir = Path(tempfile.mkdtemp(prefix="longcat_long_"))
            output = output_dir / "longcat_output.mp3"
            output.write_bytes(b"mock-mp3")
            return [object()], output
        finally:
            self.running -= 1


class ApiContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = FakeService()
        self.app = create_app(
            self.service,
            self.temp.name,
            result_ttl_hours=24,
            cleanup_interval_seconds=60,
        )
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.service.release.set()
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def add_reference(self, reference_id="voice"):
        return self.client.post(
            "/v1/references/add",
            data={"id": reference_id, "text": "Accurate transcript."},
            files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
        )

    def submit(self, step_id, **overrides):
        payload = {
            "step_id": step_id,
            "text": "Generate this sentence.",
            "language": "en",
            "reference_id": "voice",
        }
        payload.update(overrides)
        return self.client.post("/generate_audio_enhanced_async", json=payload)

    def test_reference_generation_status_download_and_idempotency(self):
        created_reference = self.add_reference()
        self.assertEqual(created_reference.status_code, 200)
        stored_audio = Path(self.temp.name) / "references" / "voice" / "audio.wav"
        with wave.open(str(stored_audio), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getframerate(), 24000)
        reused = self.add_reference()
        self.assertEqual(reused.status_code, 200)
        self.assertTrue(reused.json()["reused"])
        self.assertEqual(len(reused.json()["content_sha256"]), 64)
        response = self.submit("step-1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "pending")
        status = wait_for_status(self.client, "step-1", {"completed"})
        self.assertEqual(status["download_url"], "/download_result?step_id=step-1")
        download = self.client.get(status["download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.content, b"mock-mp3")
        duplicate = self.submit("step-1").json()
        self.assertFalse(duplicate["created"])
        self.assertEqual(len(self.service.calls), 1)
        self.assertEqual(self.service.calls[0]["speech_rate"], 1.1)
        self.assertEqual(self.service.calls[0]["target_seconds"], 24)
        self.assertEqual(self.service.calls[0]["max_seconds"], 30)
        self.assertEqual(self.service.calls[0]["steps"], 24)

    def test_null_speech_rate_uses_shared_default(self):
        self.add_reference()
        response = self.submit("null-speed", speech_rate=None)
        self.assertEqual(response.status_code, 200)
        wait_for_status(self.client, "null-speed", {"completed"})

        self.assertEqual(self.service.calls[0]["speech_rate"], 1.1)

    def test_fifo_single_worker_and_pending_cancel(self):
        self.add_reference()
        self.service.block = True
        self.submit("first")
        self.assertTrue(self.service.started.wait(1))
        self.submit("second")
        cancelled = self.client.post("/stop_async_task/second")
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        recreated = self.submit("second")
        self.assertTrue(recreated.json()["created"])
        self.service.release.set()
        wait_for_status(self.client, "first", {"completed"})
        wait_for_status(self.client, "second", {"completed"})
        self.assertEqual(self.service.max_running, 1)
        self.assertEqual(len(self.service.calls), 2)

    def test_processing_task_cancels_at_service_boundary(self):
        self.add_reference()
        self.service.block = True
        self.submit("running")
        self.assertTrue(self.service.started.wait(1))
        response = self.client.post("/stop_async_task/running")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["cancel_requested"])
        status = wait_for_status(self.client, "running", {"cancelled"})
        self.assertEqual(status["status"], "cancelled")

    def test_health_and_validation(self):
        health = self.client.get("/v1/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["model_loaded"])
        self.assertEqual(self.submit("missing").status_code, 404)
        self.add_reference()
        invalid = self.submit("bad/id")
        self.assertEqual(invalid.status_code, 422)

        self.service.is_ready = False
        unavailable = self.client.get("/v1/health")
        self.assertEqual(unavailable.status_code, 503)
        self.assertFalse(unavailable.json()["model_loaded"])
        rejected = self.submit("model-unavailable")
        self.assertEqual(rejected.status_code, 503)

    def test_reference_conflict_and_invalid_audio(self):
        self.assertEqual(self.add_reference().status_code, 200)
        conflict = self.client.post(
            "/v1/references/add",
            data={"id": "voice", "text": "Accurate transcript."},
            files={"audio": ("sample.wav", wav_bytes(sample=1), "audio/wav")},
        )
        self.assertEqual(conflict.status_code, 409)
        invalid = self.client.post(
            "/v1/references/add",
            data={"id": "broken", "text": "Transcript."},
            files={"audio": ("sample.bin", b"not-audio", "application/octet-stream")},
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertFalse(
            (Path(self.temp.name) / "references" / "broken" / "metadata.json").exists()
        )

    def test_reference_upload_size_and_duration_limits(self):
        with patch("api.app.MAX_REFERENCE_UPLOAD_BYTES", 10):
            too_large = self.client.post(
                "/v1/references/add",
                data={"id": "large", "text": "Transcript."},
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )
        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(list((Path(self.temp.name) / "staging").iterdir()), [])

        with patch("api.task_manager.MAX_REFERENCE_DURATION_SECONDS", 0.05):
            too_long = self.client.post(
                "/v1/references/add",
                data={"id": "long", "text": "Transcript."},
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )
        self.assertEqual(too_long.status_code, 422)
        self.assertFalse(
            (Path(self.temp.name) / "references" / "long" / "metadata.json").exists()
        )

    def test_normalized_text_limit_is_checked_before_queueing(self):
        self.add_reference()
        with patch("api.task_manager.MAX_GENERATION_TEXT_CHARS", 5):
            response = self.submit("expanded", text="123")
        self.assertEqual(response.status_code, 422)
        self.assertIsNone(self.app.state.task_manager.get("expanded"))

    def test_portable_resource_ids_are_rejected_by_schema(self):
        self.add_reference()
        for step_id in ("bad/id", r"bad\\id", "a..b", "trailing.", "NUL", "CON.json"):
            with self.subTest(step_id=step_id):
                self.assertEqual(self.submit(step_id).status_code, 422)

    def test_download_rejects_tampered_result_path(self):
        self.add_reference()
        self.submit("safe-step")
        wait_for_status(self.client, "safe-step", {"completed"})
        outside = Path(self.temp.name).parent / "outside-secret.txt"
        outside.write_text("secret", encoding="utf-8")
        try:
            task = self.app.state.task_manager.get("safe-step")
            task.result_path = str(outside)
            response = self.client.get(
                "/download_result", params={"step_id": "safe-step"}
            )
            self.assertEqual(response.status_code, 410)
            self.assertNotIn(b"secret", response.content)
        finally:
            outside.unlink(missing_ok=True)


class TaskPersistenceTest(unittest.TestCase):
    def test_restart_marks_active_failed_and_keeps_completed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = PersistentTaskManager(FakeService(), directory, start_worker=False)
            now = datetime.now(timezone.utc)
            pending = TaskRecord("pending", {"reference_id": "voice"})
            result = Path(directory) / "results" / "done.mp3"
            result.write_bytes(b"done")
            completed = TaskRecord(
                "done",
                {"reference_id": "voice"},
                status="completed",
                completed_at=now,
                result_path=str(result),
            )
            manager._tasks = {"pending": pending, "done": completed}
            manager._persist(pending)
            manager._persist(completed)
            restarted = PersistentTaskManager(FakeService(), directory, start_worker=False)
            self.assertEqual(restarted.get("pending").status, "failed")
            self.assertEqual(restarted.get("done").status, "completed")
            self.assertTrue(result.is_file())

    def test_restart_rejects_completed_result_outside_results_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = PersistentTaskManager(FakeService(), directory, start_worker=False)
            outside = Path(directory).parent / "longcat-outside.txt"
            outside.write_text("do not expose", encoding="utf-8")
            try:
                task = TaskRecord(
                    "unsafe",
                    {"reference_id": "voice"},
                    status="completed",
                    completed_at=datetime.now(timezone.utc),
                    result_path=str(outside),
                )
                manifest = manager.tasks_dir / "unsafe.json"
                manifest.write_text(json.dumps(task.to_dict()), encoding="utf-8")
                restarted = PersistentTaskManager(
                    FakeService(), directory, start_worker=False
                )
                restored = restarted.get("unsafe")
                self.assertEqual(restored.status, "failed")
                self.assertIsNone(restored.result_path)
                self.assertTrue(outside.is_file())
            finally:
                outside.unlink(missing_ok=True)

    def test_cleanup_uses_terminal_completion_time_only(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = PersistentTaskManager(
                FakeService(), directory, result_ttl_hours=24, start_worker=False
            )
            old = datetime.now(timezone.utc) - timedelta(hours=25)
            active = TaskRecord("active", {}, status="processing", created_at=old)
            expired = TaskRecord(
                "expired", {}, status="failed", created_at=old, completed_at=old
            )
            manager._tasks = {"active": active, "expired": expired}
            manager._persist(active)
            manager._persist(expired)
            removed = manager.cleanup_expired()
            self.assertEqual(removed, ["expired"])
            self.assertIsNotNone(manager.get("active"))

    def test_pending_limit_counts_only_pending_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = PersistentTaskManager(
                FakeService(), directory, max_pending_tasks=2, start_worker=False
            )
            manager.add_reference("voice", wav_bytes(), "voice.wav", "Transcript.")
            request = {
                "text": "Hello.",
                "language": "en",
                "reference_id": "voice",
            }
            manager.submit({**request, "step_id": "one"})
            manager.submit({**request, "step_id": "two"})
            with self.assertRaisesRegex(RuntimeError, "pending task limit"):
                manager.submit({**request, "step_id": "three"})
            duplicate, created = manager.submit({**request, "step_id": "one"})
            self.assertFalse(created)
            self.assertEqual(duplicate.step_id, "one")
            manager.cancel("two")
            manager.submit({**request, "step_id": "three"})
            self.assertEqual(manager.queue_size, 2)


if __name__ == "__main__":
    unittest.main()
