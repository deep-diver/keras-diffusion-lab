"""Structured event logging for remote training monitoring.

Model-agnostic event log that records training events as JSONL.
Supports loss, health, snapshot, and checkpoint events.
"""

import json
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    """In-memory structured event log for training monitoring."""

    def __init__(self):
        self.events = []

    def log_loss(self, step, loss, ema_loss=None, **extra):
        event = {
            "type": "loss",
            "step": step,
            "timestamp": _now_iso(),
            "loss": float(loss),
        }
        if ema_loss is not None:
            event["ema_loss"] = float(ema_loss)
        event.update(extra)
        self.events.append(event)

    def log_health(self, step, grad_norm=None, nan_detected=False, **extra):
        event = {
            "type": "health",
            "step": step,
            "timestamp": _now_iso(),
            "nan_detected": bool(nan_detected),
        }
        if grad_norm is not None:
            event["grad_norm"] = float(grad_norm)
        event.update(extra)
        self.events.append(event)

    def log_snapshot(self, step, gcs_path, loss=None):
        self.events.append({
            "type": "snapshot",
            "step": step,
            "timestamp": _now_iso(),
            "gcs_path": gcs_path,
            "loss": float(loss) if loss is not None else None,
        })

    def log_checkpoint(self, step, gcs_path=None, has_ema=False):
        self.events.append({
            "type": "checkpoint",
            "step": step,
            "timestamp": _now_iso(),
            "gcs_path": gcs_path,
            "has_ema": has_ema,
        })

    def to_jsonl(self):
        return "\n".join(json.dumps(e) for e in self.events)

    def to_json(self):
        return json.dumps(self.events, indent=2)

    def upload(self, gcs_path):
        """Upload event log to GCS."""
        from diffusion_harness.utils.gcs import upload_bytes
        upload_bytes(self.to_jsonl().encode("utf-8"), gcs_path)

    @property
    def loss_events(self):
        return [e for e in self.events if e["type"] == "loss"]

    @property
    def health_events(self):
        return [e for e in self.events if e["type"] == "health"]

    @property
    def last_loss(self):
        loss_evts = self.loss_events
        return loss_evts[-1]["loss"] if loss_evts else None


def parse_jsonl(text):
    """Parse a JSONL string into a list of event dicts."""
    events = []
    for line in text.strip().split("\n"):
        if line:
            events.append(json.loads(line))
    return events
