#!/usr/bin/env python3
"""Upload an image through the Storage Manager managed-upload API."""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests


STORAGE_MANAGER_URL = "http://192.168.1.199:8011"
IMAGE_PATH = ""
DEVICE_ID = ""
SERVICE_TOKEN = "st_5e5c88eea90f2a81b1a8100f9ab7a85b"
IS_EXTERNAL = os.environ.get("STORAGE_MANAGE_IS_EXTERNAL", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ASSET_TYPE = "snapshot"
EXTRA_DATA = {}
KEEP_FILE = False
# Keep reservation/publish responsive while serializing storage uploads in one worker.
_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minio-upload")


def _normalize_upload_url(upload_url: str, storage_manager_url: str) -> tuple[str, str | None]:
    """Connect to the host IP while preserving the presigned Host header."""
    parsed = urlsplit(upload_url)
    if (parsed.hostname or "").lower() not in {"storage-minio", "minio"}:
        return upload_url, None

    manager_host = urlsplit(storage_manager_url).hostname
    if not manager_host:
        return upload_url, None
    port = parsed.port or 9000
    netloc = f"{manager_host}:{port}"
    normalized_url = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    return normalized_url, parsed.netloc


def managed_upload_flow(
    image_path=None,
    device_id=None,
    service_token=None,
    asset_type=None,
    extra_data=None,
    keep_file=None,
    storage_manager_url=None,
    is_external=None,
    request_session=requests,
)-> str | None:
    """Reserve an asset and schedule the MinIO upload in the background.

    If no service token is configured, return no asset id. Kafka must only
    receive asset ids issued by Storage Manager.
    """
    image_path = IMAGE_PATH if image_path is None else image_path
    device_id = DEVICE_ID if device_id is None else device_id
    service_token = SERVICE_TOKEN if service_token is None else service_token
    asset_type = ASSET_TYPE if asset_type is None else asset_type
    extra_data = EXTRA_DATA if extra_data is None else extra_data
    keep_file = KEEP_FILE if keep_file is None else keep_file
    storage_manager_url = STORAGE_MANAGER_URL if storage_manager_url is None else storage_manager_url
    is_external = IS_EXTERNAL if is_external is None else bool(is_external)
    path = Path(image_path).expanduser().resolve()
    if not image_path or not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not device_id:
        raise ValueError("device_id is required")
    if not service_token:
        logging.warning("STORAGE_MANAGE_TOKEN is not set; skipping storage upload for path=%s", path)
        return None

    file_size = path.stat().st_size
    filename = path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    codec = content_type.split("/")[-1]
    metadata = extra_data if isinstance(extra_data, dict) else {}
    headers = {"X-Service-Token": service_token, "Content-Type": "application/json"}
    uploads_root = f"{storage_manager_url.rstrip('/')}/api/storage/uploads"
    reserve_payload = {
        "device_id": device_id,
        "asset_type": asset_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "file_size": file_size,
        "codec": codec,
        "extra_data": metadata,
        "is_external": is_external,
    }

    response = request_session.post(f"{uploads_root}/reserve", json=reserve_payload, headers=headers, timeout=30)
    response.raise_for_status()
    reservation = response.json()
    asset_id = reservation.get("asset_id")
    upload_url = reservation.get("upload_url")
    if not asset_id or reservation.get("method") != "PUT" or not upload_url:
        raise RuntimeError("Storage reservation did not return a valid PUT upload URL")

    upload_url, upload_host_header = _normalize_upload_url(upload_url, storage_manager_url)

    logging.info("Storage asset reserved: asset_id=%s path=%s", asset_id, path)

    _UPLOAD_EXECUTOR.submit(
        _upload_and_complete,
        path,
        upload_url,
        upload_host_header,
        uploads_root,
        asset_id,
        file_size,
        codec,
        metadata,
        headers,
        keep_file,
        request_session,
    )
    logging.info("Storage upload scheduled: asset_id=%s path=%s", asset_id, path)
    return asset_id


def _upload_and_complete(
    path,
    upload_url,
    upload_host_header,
    uploads_root,
    asset_id,
    file_size,
    codec,
    metadata,
    headers,
    keep_file,
    request_session,
):
    """Upload a reserved file and complete it in the background."""
    upload_stage = "PUT upload"
    try:
        logging.info("Storage upload started: asset_id=%s path=%s", asset_id, path)
        with path.open("rb") as image_file:
            upload_headers = {
                "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            }
            if upload_host_header:
                upload_headers["Host"] = upload_host_header
            response = request_session.put(
                upload_url,
                data=image_file,
                headers=upload_headers,
                timeout=600,
        )
        response.raise_for_status()

        upload_stage = "complete"
        response = request_session.post(
            f"{uploads_root}/{asset_id}/complete",
            json={"file_size": file_size, "codec": codec, "extra_data": metadata},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        logging.info("Storage upload completed: asset_id=%s path=%s", asset_id, path)
        return asset_id
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        body = (getattr(response, "text", "") or "").strip().replace("\n", " ")[:500]
        logging.error(
            "Storage %s failed: asset_id=%s status=%s response=%s error=%s",
            upload_stage,
            asset_id,
            status,
            body,
            exc,
        )
        logging.exception("Background MinIO upload failed for asset_id=%s", asset_id)
        raise
    except Exception:
        logging.exception(
            "Storage %s failed: asset_id=%s",
            upload_stage,
            asset_id,
        )
        raise
    finally:
        if not keep_file and path.exists():
            try:
                path.unlink()
            except OSError:
                logging.exception("Could not remove local file after upload: %s", path)
