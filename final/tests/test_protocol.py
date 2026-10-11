from datetime import datetime, timedelta, timezone

import pytest

from edgeface.protocol import Cooldown, decode, make_command, safe_name, validate_command


def test_fresh_command_and_expiry():
    command = make_command("start", "pi-test")
    assert validate_command(command, 60, "pi-test") == command
    command["timestamp"] = (datetime.now(timezone.utc)-timedelta(seconds=61)).isoformat()
    with pytest.raises(ValueError, match="Expired"):
        validate_command(command, 60, "pi-test")


@pytest.mark.parametrize("name", ["../outside", "/tmp/face", "", ".", "x\nsecret", "a"*65])
def test_names_cannot_escape_dataset(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_unknown_requires_identity_and_recorded_consent():
    command = make_command("collect_unknown", "pi-test", consent=True, identity="participant-a")
    validate_command(command, 60, "pi-test")
    command.pop("identity")
    with pytest.raises(ValueError):
        validate_command(command, 60, "pi-test")
    command["identity"] = "participant-a"
    command["consent"] = False
    with pytest.raises(ValueError, match="consent"):
        validate_command(command, 60, "pi-test")


def test_payload_limits_and_device_selection():
    with pytest.raises(ValueError):
        decode(b"[]")
    with pytest.raises(ValueError):
        decode(b"a"*16385)
    with pytest.raises(ValueError, match="device"):
        validate_command(make_command("stop", "other"), 60, "pi-test")


def test_cooldown_is_per_person():
    cooldown = Cooldown(30)
    assert cooldown.allow("son", 0)
    assert not cooldown.allow("son", 29)
    assert cooldown.allow("Unknown", 29)
    assert cooldown.allow("son", 30)
