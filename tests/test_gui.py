import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

from ins.config import config_to_mapping, load_config
from ins.gui import InsGuiServer, _existing_gui_server


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _get_json(url: str) -> dict:
    with urlopen(url) as response:  # noqa: S310 - localhost test server
        return json.loads(response.read())


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:  # noqa: S310 - localhost test server
        return json.loads(response.read())


def test_gui_serves_templates_validated_toml_and_result_artifacts(tmp_path: Path) -> None:
    server = InsGuiServer(("127.0.0.1", 0), REPOSITORY_ROOT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        assert _existing_gui_server(base_url)
        with urlopen(base_url) as response:  # noqa: S310 - localhost test server
            assert b"INS Monte Carlo Studio" in response.read()

        templates = _get_json(f"{base_url}/api/templates")
        assert templates["templates"][0]["id"] == "figure_eight"
        assert {template["id"] for template in templates["templates"]} >= {"figure_eight", "circle"}

        config = load_config(REPOSITORY_ROOT / "configs" / "figure_eight.toml")
        mapping = config_to_mapping(config)
        response = _post_json(f"{base_url}/api/config", mapping)
        assert response["filename"].endswith(".toml")
        assert "[sensors.gyro]" in response["toml"]

        mapping["simulation"].update(
            {"runs": 1, "duration_s": 1.0, "output_dir": str(tmp_path / "gui_result")}
        )
        job_id = _post_json(f"{base_url}/api/run", mapping)["job_id"]
        for _ in range(100):
            job = _get_json(f"{base_url}/api/jobs/{job_id}")
            if job["state"] in {"completed", "failed"}:
                break
            time.sleep(0.1)

        assert job["state"] == "completed", job.get("error")
        with urlopen(f"{base_url}{job['report_png_url']}") as response:  # noqa: S310 - localhost test server
            assert response.headers["Content-Type"] == "image/png"
            assert len(response.read()) > 1_000
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
