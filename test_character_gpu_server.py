"""Contract tests for the loopback GPU job service using a tiny local worker."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
from io import BytesIO
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest

from PIL import Image

from character_gpu_server import CharacterHTTPServer, JobManager, make_handler


def png() -> bytes:
    from io import BytesIO
    data = BytesIO()
    Image.new("RGB", (2, 2), "red").save(data, format="PNG")
    return data.getvalue()


class CharacterGPUServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.worker = root / "worker.py"
        self.worker.write_text(
            "import argparse, json, struct, time\n"
            "p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output'); a=p.parse_args()\n"
            "assert open(a.input,'rb').read(8) == b'\\x89PNG\\r\\n\\x1a\\n'\n"
            "if 'slow' in a.output: time.sleep(3)\n"
            "doc=b'{\"asset\":{\"version\":\"2.0\"}}'; doc+=b' ' * (-len(doc)%4)\n"
            "blob=struct.pack('<4sII',b'glTF',2,20+len(doc))+struct.pack('<I4s',len(doc),b'JSON')+doc\n"
            "open(a.output,'wb').write(blob)\n"
        )
        self.valid_worker_source = self.worker.read_text()
        self.manager = JobManager(jobs_dir=root / "jobs", worker_python=Path(sys.executable),
                                  worker_script=self.worker)
        self.server = self.start_server(self.manager)

    def start_server(self, manager):
        server = CharacterHTTPServer(("127.0.0.1", 0), make_handler(manager, "secret"))
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def request(self, method, path, body=None, authorized=True, server=None, headers=None):
        target = server or self.server
        request_headers = {"Authorization": "Bearer secret"} if authorized else {}
        request_headers.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", target.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def await_status(self, job_id, desired, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, _, body = self.request("GET", f"/jobs/{job_id}")
            self.assertEqual(status, 200)
            value = json.loads(body)
            if value["status"] == desired:
                return value
            time.sleep(0.02)
        self.fail(f"Job did not reach {desired}")

    def await_idle(self):
        deadline = time.monotonic() + 5
        while self.manager.health()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.manager.health()["busy"])

    def test_auth_png_job_and_result(self):
        self.assertEqual(self.request("GET", "/health", authorized=False)[0], 401)
        status, _, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(health), {"ready": True, "busy": False, "engine": "TRELLIS"})
        self.assertEqual(self.request("POST", "/jobs", b"not png")[0], 400)
        self.assertEqual(self.request("POST", "/jobs", b"", headers={
            "Content-Length": str(10 * 1024 * 1024 + 1)})[0], 413)
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        job_id = json.loads(body)["id"]
        self.assertEqual(len(job_id), 32)
        self.assertEqual(self.request("GET", f"/jobs/{job_id}/result")[0], 409)
        self.await_status(job_id, "succeeded")
        status, kind, glb = self.request("GET", f"/jobs/{job_id}/result")
        self.assertEqual((status, kind), (200, "model/gltf-binary"))
        self.assertEqual(glb[:4], b"glTF")
        self.assertEqual(struct.unpack_from("<I", glb, 8)[0], len(glb))
        self.assertEqual(self.request("GET", "/jobs/" + "f" * 32)[0], 404)

    def test_busy_cancel_and_sanitized_failure(self):
        # Replace the script before dispatch to hold the first job in generation.
        source = self.worker.read_text().replace("if 'slow' in a.output: time.sleep(3)", "time.sleep(3)")
        self.worker.write_text(source)
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        job_id = json.loads(body)["id"]
        self.await_status(job_id, "generating")
        with (self.manager.jobs[job_id].directory / "worker.log").open("ab") as log:
            log.write(b'{"stage":"generating"}\n')
            log.write(b'{"stage":"private error details"}\n')
        snapshot = json.loads(self.request("GET", f"/jobs/{job_id}")[2])
        self.assertEqual(snapshot["stage"], "generating")
        self.assertNotIn("private", json.dumps(snapshot))
        self.assertEqual(self.request("POST", "/jobs", png())[0], 409)
        status, _, body = self.request("DELETE", f"/jobs/{job_id}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "cancelled")
        self.await_status(job_id, "cancelled")
        deadline = time.monotonic() + 5
        while self.manager.health()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.manager.health()["busy"])

        self.worker.write_text("raise RuntimeError('private worker secret')\n")
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        failed = self.await_status(json.loads(body)["id"], "failed")
        self.assertEqual(failed["error"], "Worker failed")
        self.assertNotIn("secret", json.dumps(failed))

    def test_capacity_timeout_reports_only_safe_resource_error(self):
        self.worker.write_text("print('{\"stage\":\"waiting_gpu_memory\"}', flush=True)\n"
                               "raise RuntimeError('private capacity details')\n")
        status, _, body = self.request('POST', '/jobs', png())
        self.assertEqual(status, 202)
        result = self.await_status(json.loads(body)['id'], 'failed')
        self.assertEqual(result['error'], 'GPU memory busy')
        self.assertNotIn('private', json.dumps(result))

    def test_cleanup_only_tracks_own_directories(self):
        other = self.manager.jobs_dir / "unrelated"
        other.mkdir()
        (other / "keep").write_text("keep")
        self.manager.max_retained = 1
        for _ in range(2):
            status, _, body = self.request("POST", "/jobs", png())
            self.assertEqual(status, 202)
            self.await_status(json.loads(body)["id"], "succeeded")
            deadline = time.monotonic() + 5
            while self.manager.health()["busy"] and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertTrue((other / "keep").exists())
        self.assertEqual(len(self.manager.jobs), 1)

    def test_virtualenv_executable_path_and_ready_marker(self):
        root = Path(self.temp.name)
        link = root / "python-link"
        link.symlink_to(sys.executable)
        marker = root / "model-ready"
        manager = JobManager(jobs_dir=root / "marked-jobs", worker_python=link,
                             worker_script=self.worker, ready_file=marker)
        self.assertEqual(manager.worker_python, link)
        self.assertFalse(manager.health()["ready"])
        server = self.start_server(manager)
        self.assertEqual(self.request("POST", "/jobs", png(), server=server)[0], 503)
        marker.touch()
        self.assertTrue(manager.health()["ready"])
        status, _, body = self.request("POST", "/jobs", png(), server=server)
        self.assertEqual(status, 202)
        job_id = json.loads(body)["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, _, body = self.request("GET", f"/jobs/{job_id}", server=server)
            if status == 200 and json.loads(body)["status"] == "succeeded":
                break
            time.sleep(0.02)
        else:
            self.fail("Marked worker job did not complete")

    def test_concurrent_http_submissions_and_process_cancellation(self):
        self.worker.write_text(self.valid_worker_source.replace(
            "if 'slow' in a.output: time.sleep(3)", "time.sleep(10)"))
        with ThreadPoolExecutor(max_workers=12) as pool:
            responses = list(pool.map(lambda _: self.request("POST", "/jobs", png()), range(12)))
        self.assertEqual([response[0] for response in responses].count(202), 1)
        self.assertEqual([response[0] for response in responses].count(409), 11)
        job_id = json.loads(next(body for code, _, body in responses if code == 202))["id"]
        self.await_status(job_id, "generating")
        process = self.manager.jobs[job_id].process
        self.assertIsNotNone(process)
        self.assertEqual(self.request("DELETE", f"/jobs/{job_id}")[0], 200)
        self.await_status(job_id, "cancelled")
        deadline = time.monotonic() + 5
        while self.manager.health()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.manager.health()["busy"])
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.request("GET", f"/jobs/{job_id}/result")[0], 409)

    def test_invalid_glb_and_timeout_release_queue(self):
        self.worker.write_text(
            "import argparse\n"
            "p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output'); a=p.parse_args()\n"
            "open(a.output,'wb').write(b'not a GLB')\n"
        )
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        failed = self.await_status(json.loads(body)["id"], "failed")
        self.assertEqual(failed["error"], "Worker returned an invalid GLB")
        deadline = time.monotonic() + 5
        while self.manager.health()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)

        self.manager.timeout = 0.15
        self.worker.write_text("import time\ntime.sleep(10)\n")
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        failed = self.await_status(json.loads(body)["id"], "failed")
        self.assertEqual(failed["error"], "Generation timed out")
        deadline = time.monotonic() + 5
        while self.manager.health()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.manager.health()["busy"])

        self.worker.write_text(self.valid_worker_source)
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        self.await_status(json.loads(body)["id"], "succeeded")

    def test_result_ttl_removes_finished_job_and_owned_directory(self):
        self.manager.result_ttl = 0.12
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        job_id = json.loads(body)["id"]
        self.await_status(job_id, "succeeded")
        directory = self.manager.jobs_dir / job_id
        self.assertTrue(directory.is_dir())
        time.sleep(0.15)
        self.assertEqual(self.request("GET", f"/jobs/{job_id}")[0], 404)
        self.assertEqual(self.request("GET", f"/jobs/{job_id}/result")[0], 404)
        self.assertFalse(directory.exists())

    def test_auth_is_required_for_every_job_route(self):
        self.assertEqual(self.request("POST", "/jobs", png(), authorized=False)[0], 401)
        status, _, body = self.request("POST", "/jobs", png())
        self.assertEqual(status, 202)
        job_id = json.loads(body)["id"]
        for method, route in (("GET", f"/jobs/{job_id}"),
                              ("GET", f"/jobs/{job_id}/result"),
                              ("DELETE", f"/jobs/{job_id}")):
            with self.subTest(method=method, route=route):
                self.assertEqual(self.request(method, route, authorized=False)[0], 401)
        self.assertEqual(self.request("GET", "/jobs/../health")[0], 404)
        self.await_status(job_id, "succeeded")

    def test_malformed_png_and_glb_framing_over_http(self):
        valid_png = png()
        for bad_png in (valid_png[:8], valid_png[:-15], b"\x89PNG\r\n\x1a\n" + b"not a PNG"):
            with self.subTest(png_size=len(bad_png)):
                self.assertEqual(self.request("POST", "/jobs", bad_png)[0], 400)

        document = b'{"asset":{"version":"2.0"}}'
        document += b" " * (-len(document) % 4)
        good = (struct.pack("<4sII", b"glTF", 2, 20 + len(document))
                + struct.pack("<I4s", len(document), b"JSON") + document)
        malformed = {
            "wrong_declared_length": good[:8] + struct.pack("<I", len(good) + 4) + good[12:],
            "missing_asset": (struct.pack("<4sII", b"glTF", 2, 24)
                              + struct.pack("<I4s", 4, b"JSON") + b"{}  "),
            "bad_chunk_type": good[:16] + b"BIN\0" + good[20:],
        }
        for name, blob in malformed.items():
            with self.subTest(glb=name):
                self.worker.write_text(
                    "import argparse\n"
                    "p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output'); a=p.parse_args()\n"
                    f"open(a.output,'wb').write(bytes.fromhex('{blob.hex()}'))\n"
                )
                status, _, body = self.request("POST", "/jobs", valid_png)
                self.assertEqual(status, 202)
                failed = self.await_status(json.loads(body)["id"], "failed")
                self.assertEqual(failed["error"], "Worker returned an invalid GLB")
                self.await_idle()


if __name__ == "__main__":
    unittest.main()
