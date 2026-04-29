"""Tests for monitoring."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import json

from diffusion_harness.monitoring import EventLog, parse_jsonl


def test_empty_event_log():
    log = EventLog()
    assert len(log.events) == 0
    assert log.last_loss is None


def test_log_loss():
    log = EventLog()
    log.log_loss(step=10, loss=0.5)
    log.log_loss(step=20, loss=0.3, ema_loss=0.4)
    assert len(log.loss_events) == 2
    assert log.last_loss == 0.3


def test_log_health():
    log = EventLog()
    log.log_health(step=100, grad_norm=1.5, nan_detected=False)
    assert len(log.health_events) == 1
    assert log.health_events[0]["grad_norm"] == 1.5


def test_to_jsonl():
    log = EventLog()
    log.log_loss(step=10, loss=0.5)
    log.log_health(step=100, grad_norm=1.0)
    jsonl = log.to_jsonl()
    lines = [l for l in jsonl.strip().split("\n") if l]
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "type" in parsed
        assert "step" in parsed


def test_parse_jsonl():
    text = '{"type": "loss", "step": 10, "loss": 0.5}\n{"type": "health", "step": 20}'
    events = parse_jsonl(text)
    assert len(events) == 2
    assert events[0]["type"] == "loss"


def test_roundtrip():
    log = EventLog()
    log.log_loss(step=10, loss=0.5)
    log.log_checkpoint(step=100, has_ema=True)
    jsonl = log.to_jsonl()
    restored = parse_jsonl(jsonl)
    assert len(restored) == 2
    assert restored[0]["loss"] == 0.5
    assert restored[1]["has_ema"] is True
