"""Real CPU transitions and the manual HTTP boundary; no training is run."""

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from terra.viewer3d.server import make_server
from terra.viewer3d.session import ManualSession


class ViewerSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = ManualSession()
        cls.base_config = cls.session.config
        cls.server = make_server(session=cls.session, port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.session.config = self.base_config
        self.session.reset()

    def request(self, path, data=None, headers=None):
        body = json.dumps(data).encode() if data is not None else None
        request = Request(
            self.url + path,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            response = urlopen(request, timeout=90)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_real_dig_slew_dump_conserves_soil(self):
        first = self.session.recorder.frames[0]
        before = np.array(first["maps"]["action"])
        dig = self.session.step(6)
        self.assertGreater(dig["agents"][0]["loaded"], 0)
        self.assertTrue(np.any(np.array(dig["maps"]["action"]) < before))
        self.assertEqual(dig["actor_id"], 0)
        for _ in range(6):
            self.session.step(4)
        dump = self.session.step(6)
        self.assertEqual(dump["agents"][0]["loaded"], 0)
        self.assertTrue(
            np.any(np.array(dump["maps"]["action"]) > np.array(dig["maps"]["action"]))
        )
        self.assertEqual(first["maps"]["action"], before.tolist())
        self.assertGreater(np.max(dump["maps"]["action"]), 1)
        for frame in self.session.recorder.frames:
            total = np.sum(frame["maps"]["action"]) + sum(
                a["loaded"] for a in frame["agents"]
            )
            self.assertEqual(total, np.sum(before))

    def test_real_motion_orientation_and_failed_action(self):
        original = self.session.recorder.frames[0]["agents"][0]["position"]
        moved = self.session.step(1)
        self.assertEqual(
            moved["agents"][0]["position"],
            [original[0], original[1] - self.base_config.agent.move_tiles],
        )
        self.session.step(0)
        rotated = self.session.step(2)
        expected = (
            2
            * np.pi
            * (self.base_config.agent.angles_base - 1)
            / self.base_config.agent.angles_base
        )
        self.assertAlmostEqual(rotated["agents"][0]["base_yaw"], expected, places=5)
        self.session.reset()
        dig = self.session.step(6)
        failed = self.session.step(6)  # A loaded bucket cannot dump into this hole.
        self.assertEqual(failed["maps"]["action"], dig["maps"]["action"])
        self.assertEqual(failed["agents"], dig["agents"])

    def test_terminal_state_retained_and_step_rejected(self):
        self.session.config = self.base_config._replace(max_steps_in_episode=1)
        self.session.reset()
        terminal = self.session.step(7)
        self.assertTrue(terminal["done"])
        self.assertFalse(terminal["task_done"])
        self.assertEqual(terminal["step"], 1)
        with self.assertRaisesRegex(RuntimeError, "ended"):
            self.session.step(7)
        self.assertEqual(len(self.session.recorder.frames), 2)
        status, body = self.request("/api/action", {"action": 7})
        self.assertEqual(status, 409)
        self.assertIn("ended", body["error"])

    def test_http_play_export_reset_and_invalid_input(self):
        status, initial = self.request("/api/session")
        self.assertEqual(status, 200)
        self.assertEqual(initial["mode"], "manual")
        for action in (True, -1, 8, 2.5, "6", None):
            status, body = self.request("/api/action", {"action": action})
            self.assertEqual(status, 400)
            self.assertIn("action", body["error"])
        self.assertEqual(len(self.session.recorder.frames), 1)
        status, changed = self.request("/api/action", {"action": 6})
        self.assertEqual(status, 200)
        self.assertGreater(changed["frame"]["agents"][0]["loaded"], 0)
        status, replay = self.request("/api/replay")
        self.assertEqual(status, 200)
        self.assertEqual(len(replay["frames"]), 2)
        self.assertEqual(replay["frames"][-1], changed["frame"])
        status, reset = self.request("/api/reset", {})
        self.assertEqual(status, 200)
        self.assertEqual(reset["replay"]["frames"], initial["replay"]["frames"])

    def test_http_rejects_cross_origin_and_unknown_routes(self):
        status, _ = self.request(
            "/api/action", {"action": 6}, {"Origin": "https://elsewhere.example"}
        )
        self.assertEqual(status, 403)
        status, _ = self.request("/api/session", headers={"Host": "elsewhere.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("/../session.py")
        self.assertEqual(status, 404)
        self.assertEqual(len(self.session.recorder.frames), 1)


if __name__ == "__main__":
    unittest.main()
