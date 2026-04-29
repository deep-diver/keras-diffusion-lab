"""GCS utility helpers for artifact persistence.

Generic upload/download functions for files, bytes, JSON, and numpy arrays.
Used for checkpoint persistence, event log upload, and artifact sync during
remote TPU training.
"""

import json
import os
import io
import numpy as np


def _get_client():
    from google.cloud import storage
    return storage.Client()


def _parse_gcs_path(gcs_path):
    """Parse gs://bucket/path into (bucket_name, blob_path)."""
    assert gcs_path.startswith("gs://"), f"Not a GCS path: {gcs_path}"
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_path


def upload_file(local_path, gcs_path):
    """Upload a local file to GCS."""
    bucket_name, blob_path = _parse_gcs_path(gcs_path)
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def upload_bytes(data, gcs_path):
    """Upload raw bytes to GCS."""
    bucket_name, blob_path = _parse_gcs_path(gcs_path)
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type="application/octet-stream")


def download_file(gcs_path, local_path):
    """Download a file from GCS. Returns True if successful."""
    try:
        bucket_name, blob_path = _parse_gcs_path(gcs_path)
        client = _get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return False
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return True
    except Exception:
        return False


def download_bytes(gcs_path):
    """Download raw bytes from GCS. Returns None if not found."""
    try:
        bucket_name, blob_path = _parse_gcs_path(gcs_path)
        client = _get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    except Exception:
        return None


def upload_json(data, gcs_path):
    """Upload a JSON-serializable object to GCS."""
    upload_bytes(json.dumps(data, indent=2).encode("utf-8"), gcs_path)


def download_json(gcs_path):
    """Download and parse JSON from GCS. Returns None if not found."""
    raw = download_bytes(gcs_path)
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def upload_numpy(array, gcs_path):
    """Upload a numpy array to GCS."""
    buf = io.BytesIO()
    np.save(buf, array)
    upload_bytes(buf.getvalue(), gcs_path)


def download_numpy(gcs_path):
    """Download and load a numpy array from GCS."""
    raw = download_bytes(gcs_path)
    if raw is None:
        return None
    return np.load(io.BytesIO(raw))


def list_blobs(gcs_path, pattern=None):
    """List blobs under a GCS prefix, optionally filtering by pattern."""
    bucket_name, prefix = _parse_gcs_path(gcs_path)
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    results = [f"gs://{bucket_name}/{b.name}" for b in blobs]
    if pattern:
        results = [r for r in results if pattern in r]
    return results


def find_latest_checkpoint(gcs_dir):
    """Find the latest model checkpoint in a GCS directory.

    Looks for files matching 'model_stepN.weights.h5' and returns
    the path with the highest step number.
    """
    blobs = list_blobs(gcs_dir, pattern="model_step")
    if not blobs:
        return None

    best_step = -1
    best_path = None
    for blob_path in blobs:
        name = blob_path.split("/")[-1]
        for part in name.split("_"):
            if part.startswith("step"):
                step_str = part.replace("step", "").split(".")[0]
                try:
                    step = int(step_str)
                    if step > best_step:
                        best_step = step
                        best_path = blob_path
                except ValueError:
                    continue
    return best_path
