import asyncio
import json
from unittest.mock import AsyncMock

from edgeface.config import Config
from telegram_bot.bot import Bot


def bot(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    return Bot(Config(mqtt_password="test-only", auto_stop_known=True))


def test_known_recognition_stops_matching_session(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "idle"})
    text = asyncio.run(b.auto_stop_result("result", {"event": "recognition", "person": "son", "request_id": "session-id"}))
    command = b.dispatch.call_args.args[0]
    assert command["action"] == "stop" and command["expected_request_id"] == "session-id"
    assert "Camera stopped automatically" in text


def test_unknown_and_no_face_do_not_stop(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock()
    async def check():
        assert await b.auto_stop_result("result", {"event": "recognition", "person": "Unknown", "request_id": "id"}) == ""
        assert await b.auto_stop_result("result", {"event": "no_face"}) == ""
    asyncio.run(check())
    b.dispatch.assert_not_called()


def test_auto_stop_waits_for_actual_ack_and_handles_timeout(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(side_effect=RuntimeError("timeout"))
    text = asyncio.run(b.auto_stop_result("result", {"event": "recognition", "person": "son", "request_id": "session"}))
    assert "not confirmed" in text and "Camera stopped automatically" not in text


def test_ack_can_resolve_while_consumer_awaits_auto_stop(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    async def check():
        b.loop = asyncio.get_running_loop()
        future = b.loop.create_future()
        b.pending["request"] = future
        b.receive("status", json.dumps({"device_id": b.cfg.device_id, "event": "ack", "request_id": "request", "ok": True}).encode(), False)
        response = await asyncio.wait_for(future, 0.2)
        assert response["ok"] and b.queue.empty()
    asyncio.run(check())


def test_result_displays_full_name_but_keeps_class_id(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.display_names = {"son": "Nguyễn Hồ Trường Sơn"}
    value = {"event": "recognition", "person": "son", "confidence": 0.95,
             "device_id": "pi-test", "inference_ms": 30}
    text = b.event_text("result", value)
    assert "Person: Nguyễn Hồ Trường Sơn" in text
    assert value["person"] == "son"
    assert b.display_name("Unknown") == "Unknown"


def test_unknown_is_silent_in_telegram(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    assert b.event_text("result", {"event": "recognition", "person": "Unknown"}) is None


def test_start_shows_recognizing_status(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "recognizing"})
    message = type("Message", (), {"reply_text": AsyncMock()})()
    asyncio.run(b.execute(message, {"action": "start"}))
    message.reply_text.assert_awaited_once_with("Đang nhận diện...")
