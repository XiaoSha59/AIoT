"""Logical safety tests with explicit fake camera/model; these are NOT real demo evidence."""
import json
import time

import numpy as np
import pytest

from edgeface.config import Config
from edgeface.protocol import make_command
from raspberry_pi.agent import Agent
from raspberry_pi.camera import CameraError
from raspberry_pi.inference import ModelError


class FakeLink:
    def __init__(self):
        self.messages = []

    def publish(self, topic, value, retain=False):
        self.messages.append((topic, value, retain))
        return True


class FakeCamera:
    backend = "fake"
    opened = 0
    def read(self):
        return np.zeros((224, 224, 3), np.uint8)
    def open(self):
        self.opened += 1
        self.backend = "fake"
    def close(self):
        self.backend = None


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
    def detect(self, frame):
        return self.faces


class FakeClassifier:
    version = "test-model"
    bundle_version = "test-bundle"
    threshold = 0.75
    calls = 0
    def predict(self, crop):
        self.calls += 1
        return "Unknown", 0.55, 1.0


@pytest.fixture
def agent(tmp_path):
    cfg = Config(dataset_dir=tmp_path/"dataset", mqtt_password="test-only", no_face_after=0, cooldown=30)
    a = Agent(cfg)
    a.link = FakeLink()
    a.camera = FakeCamera()
    a.classifier = FakeClassifier()
    a.state = "recognizing"
    yield a
    a.camera.close()
    a.speaker.close()
    a.db.close()


def test_no_face_never_calls_classifier(agent):
    agent.detector = FakeDetector([])
    agent.frame()
    assert agent.classifier.calls == 0
    assert agent.link.messages[-1][1]["event"] == "no_face"


def test_multiple_faces_never_assigned_arbitrary_identity(agent):
    face = np.array([20, 20, 100, 100], np.float32)
    agent.detector = FakeDetector([face, face])
    agent.frame()
    assert agent.classifier.calls == 0
    assert agent.link.messages[-1][1]["event"] == "multiple_faces"


def test_unknown_event_and_cooldown(agent):
    agent.detector = FakeDetector([np.array([20, 20, 100, 100], np.float32)])
    agent.frame()
    agent.frame()
    events = [v for t, v, _ in agent.link.messages if t == "result"]
    assert len(events) == 1 and events[0]["person"] == "Unknown"
    assert agent.classifier.calls == 2


def test_retained_command_never_opens_camera(agent):
    command = make_command("start", agent.cfg.device_id)
    agent.on_message("command", json.dumps(command).encode(), True)
    assert agent.commands.empty() and agent.camera.opened == 0
    assert agent.link.messages[-1][1]["ok"] is False


@pytest.mark.parametrize("error,kind", [(CameraError("lost"), "camera"), (ModelError("bad model"), "model")])
def test_camera_and_model_errors_are_distinct_and_camera_closes(agent, error, kind):
    agent.fail(error)
    assert agent.state == "error" and agent.error["kind"] == kind
    assert agent.camera.backend is None


def test_request_claim_survives_agent_restart(tmp_path, monkeypatch):
    cfg = Config(dataset_dir=tmp_path/"dataset", mqtt_password="test-only")
    a = Agent(cfg)
    a.link = FakeLink()
    command = make_command("status", cfg.device_id)
    a.handle(command)
    a.speaker.close()
    a.db.close()
    b = Agent(cfg)
    b.link = FakeLink()
    b.handle(command)
    assert b.link.messages[-1][1]["duplicate"] is True
    b.speaker.close()
    b.db.close()


def test_old_recognition_cannot_stop_new_session(agent):
    agent.active_request = "new-session"
    command = make_command("stop", agent.cfg.device_id, expected_request_id="old-session")
    agent.handle(command)
    assert agent.state == "recognizing" and agent.camera.backend == "fake"
    response = next(v for t, v, retained in agent.link.messages if v.get("event") == "ack")
    assert response["stop_skipped"]


@pytest.mark.parametrize("snapshot_fails", [False, True])
def test_known_snapshot_uses_matching_frame_and_failure_preserves_result(agent, snapshot_fails):
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import Mock

    agent.cfg = replace(agent.cfg, recognition_photo_enabled=True)
    agent.link.connected = SimpleNamespace(is_set=lambda: True)
    agent.detector = FakeDetector([np.array([20, 20, 100, 100], np.float32)])
    agent.classifier.predict = lambda crop: ("son", .94, 30.0)
    agent.snapshots.add = Mock(side_effect=OSError("simulated snapshot failure") if snapshot_fails else None,
                               return_value={"id": "a"*32})
    agent.frame()
    result = [v for t, v, _ in agent.link.messages if t == "result"][-1]
    assert result["person"] == "son" and result["confidence"] == .94
    assert ("image" in result) is not snapshot_fails
    args = agent.snapshots.add.call_args.args
    assert args[0].shape == (224, 224, 3) and args[3] == .94
    assert agent.state == "recognizing"
