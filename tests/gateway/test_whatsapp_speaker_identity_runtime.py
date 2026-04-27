from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(identity_map=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True,
                extra={"sender_identity_map": identity_map or {}},
            )
        }
    )
    return runner


def _make_event(user_name="Eliran Aldoroti", sender_phone="+972547795455", text="שלום"):
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="120363427147164635@g.us",
        chat_type="group",
        user_id="207618769473605@lid",
        user_name=user_name,
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"senderPhone": sender_phone},
    )


def test_apply_whatsapp_group_speaker_context_uses_phone_mapping_for_eliran():
    runner = _make_runner({
        "+972547795455": {"name": "Eliran Aldoroti", "gender": "male", "role": "operator"}
    })
    event = _make_event(user_name="Eliran Aldoroti", sender_phone="972547795455")

    result = runner._apply_whatsapp_group_speaker_context(event, "שלום")

    assert result.startswith("[Speaker: Eliran Aldoroti | gender: male | role: operator]")
    assert result.endswith("שלום")


def test_apply_whatsapp_group_speaker_context_uses_phone_mapping_for_miriam():
    runner = _make_runner({
        "+972506705646": {"name": "מרים אלדרוטי", "gender": "female", "role": "primary"}
    })
    event = _make_event(user_name="מרים אלדרוטי", sender_phone="+972506705646")

    result = runner._apply_whatsapp_group_speaker_context(event, "מה שלומך")

    assert result.startswith("[Speaker: מרים אלדרוטי | gender: female | role: primary]")
    assert result.endswith("מה שלומך")


def test_apply_whatsapp_group_speaker_context_falls_back_to_user_name_when_unmapped():
    runner = _make_runner({})
    event = _make_event(user_name="Unknown Person", sender_phone="+972500000000")

    result = runner._apply_whatsapp_group_speaker_context(event, "hello")

    assert result == "[Speaker: Unknown Person]\nhello"
