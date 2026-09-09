import json

import pytest

from funasr_ws.protocol import (
    PartialTracker,
    ProtocolError,
    parse_control_frame,
    parse_hotwords,
    parse_language,
    parse_speech_noise_threshold,
    parse_start_frame,
)


def test_parse_start_frame_from_jarvis_shape():
    frame = parse_start_frame(
        json.dumps(
            {
                "mode": "2pass",
                "wav_name": "abc",
                "wav_format": "pcm",
                "audio_fs": 16000,
                "is_speaking": True,
                "chunk_size": [5, 10, 5],
                "itn": True,
                "svs_lang": "en",
                "svs_itn": True,
                "speech_noise_thres": 0.7,
                "hotwords": json.dumps({"Jarvis": 20, "GitHub": 20}),
            }
        )
    )
    assert frame.wav_name == "abc"
    assert frame.language == "英文"
    assert frame.hotwords == ("Jarvis", "GitHub")
    assert frame.speech_noise_threshold == 0.7
    assert frame.is_speaking is True


def test_parse_start_frame_defaults():
    frame = parse_start_frame(b"{}")
    assert frame.wav_name == "stream"
    assert frame.language == ""
    assert frame.hotwords == ()
    assert frame.speech_noise_threshold is None


@pytest.mark.parametrize(
    "value,expected", [("auto", ""), (None, ""), ("zh-CN", "中文"), ("JA", "日文")]
)
def test_parse_language(value, expected):
    assert parse_language(value) == expected


def test_parse_language_rejects_unknown():
    with pytest.raises(ProtocolError):
        parse_language("fr")


@pytest.mark.parametrize(
    "raw", ["not json", "[1,2]", '{"audio_fs": 8000}', '{"wav_format": "wav"}']
)
def test_parse_start_frame_rejects(raw):
    with pytest.raises(ProtocolError):
        parse_start_frame(raw)


@pytest.mark.parametrize("value", [1, -0.1, "0.5", True])
def test_speech_noise_threshold_rejects(value):
    with pytest.raises(ProtocolError):
        parse_speech_noise_threshold(value)


def test_hotwords_rejects_non_object():
    with pytest.raises(ProtocolError):
        parse_hotwords("[1]")


def test_control_frame_rejects_late_threshold():
    with pytest.raises(ProtocolError):
        parse_control_frame('{"speech_noise_thres": 0.5}')
    assert parse_control_frame('{"is_speaking": false}') == {"is_speaking": False}


def test_partial_tracker_emits_only_new_text():
    tracker = PartialTracker()
    assert tracker.delta("请帮我") == "请帮我"
    assert tracker.delta("请帮我把明天") == "把明天"
    # A revision of the earlier words is held back until the text grows past it.
    assert tracker.delta("请帮你把明天") == ""
    assert tracker.delta("请帮你把明天的会") == "的会"
    tracker.reset()
    assert tracker.delta("Hello") == "Hello"
