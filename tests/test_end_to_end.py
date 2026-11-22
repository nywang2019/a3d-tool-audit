from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from llm_eval.runner import load_config, run_experiment


class LocalProviderHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(content_length))
        type(self).request_count += 1
        assert body["messages"][0]["role"] == "user"
        payload = json.dumps(
            {
                "choices": [{"message": {"content": r"Work complete: \boxed{0}"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return None


def test_full_runner_with_local_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = load_config("configs/pretest.yaml")
        config["experiment"]["tasks"] = ["sum", "sorting"]
        config["experiment"]["datapoints_per_task"] = 1
        config["experiment"]["max_tokens"] = 64
        config["models"] = [config["models"][0]]
        config["models"][0].update(
            {
                "enabled": True,
                "alias": "local-fake",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "model": "local-deterministic-fake",
                "api_key_env": "LOCAL_FAKE_KEY",
                "max_retries": 0,
            }
        )
        config["budget"] = {
            "max_requests_per_model": 4,
            "max_total_requests": 4,
        }
        monkeypatch.setenv("LOCAL_FAKE_KEY", "local-secret")
        manifest = run_experiment(config, results_dir=tmp_path)
        assert manifest["completed_models"] == ["local-fake"]
        assert manifest["analysis_completed"] is True
        run_root = Path(manifest["run_root"])
        assert (run_root / "analysis" / "accuracy_summary.csv").exists()
        raw_logs = list((run_root / "local-fake" / "raw").glob("*.json"))
        assert len(raw_logs) == 2
        assert all(
            "local-secret" not in path.read_text(encoding="utf-8") for path in raw_logs
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
