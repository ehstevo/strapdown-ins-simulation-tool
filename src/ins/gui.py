"""Local browser GUI for configuring and running INS Monte Carlo studies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import errno
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import tempfile
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
import webbrowser

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ins-monte-carlo-matplotlib"))

import matplotlib

matplotlib.use("Agg")

from ins.config import (
    ConfigurationError,
    SimulationConfig,
    config_from_mapping,
    config_to_mapping,
    config_to_toml,
    load_config,
)
from ins.monte_carlo import run_monte_carlo
from ins.reporting import save_results


MAX_REQUEST_BYTES = 1_000_000
SAFE_FILENAME = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass
class SimulationJob:
    """Mutable state for one browser-initiated simulation."""

    config: SimulationConfig
    state: str = "queued"
    completed_runs: int = 0
    total_runs: int = 0
    output_dir: Path | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        """Return a JSON-safe view of the current job state."""

        with self.lock:
            payload: dict[str, Any] = {
                "id": job_id,
                "state": self.state,
                "completed_runs": self.completed_runs,
                "total_runs": self.total_runs,
            }
            if self.error is not None:
                payload["error"] = self.error
            if self.output_dir is not None:
                payload["output_dir"] = str(self.output_dir)
                payload["report_png_url"] = f"/api/jobs/{job_id}/monte_carlo_summary.png"
                payload["report_pdf_url"] = f"/api/jobs/{job_id}/monte_carlo_summary.pdf"
                payload["config_url"] = f"/api/jobs/{job_id}/config.toml"
            return payload


class InsGuiServer(ThreadingHTTPServer):
    """Threaded localhost server that owns frontend simulation jobs."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], workspace: Path):
        super().__init__(address, InsGuiRequestHandler)
        self.workspace = workspace.resolve()
        self.jobs: dict[str, SimulationJob] = {}
        self.jobs_lock = threading.Lock()
        self.next_job_number = 1

    def templates(self) -> list[dict[str, Any]]:
        """Return built-in TOML configurations as form-ready JSON mappings."""

        templates: list[dict[str, Any]] = []
        paths = sorted(
            (self.workspace / "configs").glob("*.toml"),
            key=lambda path: (path.stem != "figure_eight", path.name),
        )
        for path in paths:
            try:
                config = load_config(path)
            except ConfigurationError:
                continue
            templates.append(
                {
                    "id": path.stem,
                    "label": config.name,
                    "config": config_to_mapping(config),
                }
            )
        return templates

    def start_job(self, config: SimulationConfig) -> str:
        """Start one simulation worker and return its browser-visible job id."""

        with self.jobs_lock:
            job_id = str(self.next_job_number)
            self.next_job_number += 1
            job = SimulationJob(config=config, total_runs=config.runs)
            self.jobs[job_id] = job

        worker = threading.Thread(
            target=self._run_job,
            args=(job_id, job),
            daemon=True,
            name=f"ins-monte-carlo-{job_id}",
        )
        worker.start()
        return job_id

    def _run_job(self, job_id: str, job: SimulationJob) -> None:
        """Run a simulation off the HTTP request thread."""

        def progress(completed_runs: int, total_runs: int) -> None:
            with job.lock:
                job.completed_runs = completed_runs
                job.total_runs = total_runs

        with job.lock:
            job.state = "running"
        try:
            result = run_monte_carlo(job.config, progress_callback=progress)
            output_dir = save_results(result, job.config)
        except Exception as error:  # Surface a useful message to the local user.
            with job.lock:
                job.state = "failed"
                job.error = str(error)
            return

        with job.lock:
            job.state = "completed"
            job.completed_runs = job.total_runs
            job.output_dir = output_dir


class InsGuiRequestHandler(BaseHTTPRequestHandler):
    """Serve the single-page GUI and its local JSON API."""

    server: InsGuiServer

    def log_message(self, format: str, *args: Any) -> None:
        """Keep routine browser requests out of the terminal."""

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._serve_index()
        elif path == "/api/templates":
            self._send_json(HTTPStatus.OK, {"templates": self.server.templates()})
        elif path.startswith("/api/jobs/"):
            self._serve_job_get(path)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/shutdown":
            self._send_json(HTTPStatus.ACCEPTED, {"message": "Stopping local GUI server."})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path not in {"/api/config", "/api/run"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return

        try:
            mapping = self._read_json()
            config = config_from_mapping(mapping, default_name="gui_simulation")
        except (ConfigurationError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        if path == "/api/config":
            filename = SAFE_FILENAME.sub("_", config.name.lower()).strip("_") or "ins_simulation"
            self._send_json(
                HTTPStatus.OK,
                {"filename": f"{filename}.toml", "toml": config_to_toml(config)},
            )
            return

        job_id = self.server.start_job(config)
        self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})

    def _serve_index(self) -> None:
        index_path = Path(__file__).with_name("web") / "index.html"
        self._send_bytes(HTTPStatus.OK, index_path.read_bytes(), "text/html; charset=utf-8")

    def _serve_job_get(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 4 or not parts[3].isdigit():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown simulation job."})
            return

        job_id = parts[3]
        with self.server.jobs_lock:
            job = self.server.jobs.get(job_id)
        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown simulation job."})
            return

        if len(parts) == 4:
            self._send_json(HTTPStatus.OK, job.snapshot(job_id))
            return

        filename = parts[4]
        if filename not in {"monte_carlo_summary.png", "monte_carlo_summary.pdf", "config.toml"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown result artifact."})
            return

        with job.lock:
            output_dir = job.output_dir
        if output_dir is None:
            self._send_json(HTTPStatus.CONFLICT, {"error": "Simulation has not completed."})
            return
        artifact = output_dir / filename
        if not artifact.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Result artifact is unavailable."})
            return
        content_type = mimetypes.guess_type(str(artifact))[0] or "application/octet-stream"
        self._send_bytes(HTTPStatus.OK, artifact.read_bytes(), content_type)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("Request must contain a JSON configuration smaller than 1 MB.")
        payload = self.rfile.read(content_length)
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Configuration request must contain a JSON object.")
        return decoded

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local INS Monte Carlo configuration GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Local interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", default=8765, type=int, help="Local TCP port (default: 8765).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the default browser automatically.")
    return parser.parse_args()


def _existing_gui_server(url: str) -> bool:
    """Return whether an occupied port belongs to another local GUI instance."""

    try:
        with urlopen(f"{url}/api/templates", timeout=0.5) as response:  # noqa: S310 - localhost probe
            payload = json.loads(response.read())
    except (OSError, URLError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("templates"), list)


def main() -> None:
    """Launch the local browser GUI until interrupted with Ctrl+C."""

    arguments = _arguments()
    requested_url = f"http://{arguments.host}:{arguments.port}"
    try:
        server = InsGuiServer((arguments.host, arguments.port), Path.cwd())
    except OSError as error:
        if error.errno == errno.EADDRINUSE and _existing_gui_server(requested_url):
            print(f"INS Monte Carlo GUI is already running at {requested_url}")
            if not arguments.no_browser:
                webbrowser.open(requested_url)
            return
        raise SystemExit(
            f"Unable to start the GUI on {requested_url}: {error}. "
            "Use --port to choose another port."
        ) from error

    url = f"http://{arguments.host}:{server.server_port}"
    print(f"INS Monte Carlo GUI available at {url}")
    print("Press Ctrl+C to stop the local server.")
    if not arguments.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping INS Monte Carlo GUI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
