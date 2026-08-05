"""Command-line entry point for the LongCat AudioDiT FastAPI server."""

from __future__ import annotations

import argparse

import uvicorn

from api import create_app
from app_config import (
    DEFAULT_API_DATA_DIR,
    DEFAULT_API_PORT,
    DEFAULT_DEVICE,
    DEFAULT_HOST,
    DEFAULT_MAX_PENDING_TASKS,
    DEFAULT_MODEL_DIR,
    DEFAULT_RESULT_TTL_HOURS,
)
from services import AudioDiTService


class UnavailableAudioDiTService:
    """Keep diagnostics reachable when startup model loading fails."""

    is_ready = False

    def __init__(self, model_name: str, device: str | None, error: Exception):
        self.model_name = model_name
        self.device = device
        self.load_error = str(error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LongCat AudioDiT API")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--data_dir", default=DEFAULT_API_DATA_DIR)
    parser.add_argument(
        "--result_ttl_hours", type=float, default=DEFAULT_RESULT_TTL_HOURS
    )
    parser.add_argument(
        "--max_pending_tasks", type=int, default=DEFAULT_MAX_PENDING_TASKS
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"Loading model from {args.model_dir} on {args.device or 'auto'}...")
    try:
        service = AudioDiTService.from_pretrained(args.model_dir, args.device)
    except Exception as error:
        print(f"Model loading failed; health endpoint will report unavailable: {error}")
        service = UnavailableAudioDiTService(args.model_dir, args.device, error)
    app = create_app(
        service,
        args.data_dir,
        args.result_ttl_hours,
        max_pending_tasks=args.max_pending_tasks,
    )
    print(f"Starting API at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
