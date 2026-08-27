#!/usr/bin/env python3
"""Upload an image through the Storage Manager managed-upload API."""

import mimetypes
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests


STORAGE_MANAGER_URL = "http://192.168.1.199:8011"
IMAGE_PATH = ""
DEVICE_ID = ""
SERVICE_TOKEN = "st_5e5c88eea90f2a81b1a8100f9ab7a85b"
IS_EXTERNAL = os.environ.get("STORAGE_MANAGE_IS_EXTERNAL", True)
ASSET_TYPE = "snapshot"
EXTRA_DATA = {}
KEEP_FILE = False
_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="minio-upload")

# Import from another service like this:
# from src.scripts.upload_image_to_minio import managed_upload_flow
# asset_id = managed_upload_flow()


def managed_upload_flow(
    image_path=None, device_id=None, service_token=None, asset_type=None,
    extra_data=None, keep_file=None, storage_manager_url=None,
    is_external=None, request_session=requests,
):
    """Reserve an asset and schedule the MinIO upload in the background.

    Returns the asset ID as soon as the reservation succeeds. The upload and
    completion request continue in the module's background worker.
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
        raise ValueError("STORAGE_MANAGE_TOKEN or service_token is required")

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

    response = request_session.post(f"{uploads_root}/reserve", json=reserve_payload,
                                    headers=headers, timeout=30)
    response.raise_for_status()
    reservation = response.json()
    asset_id = reservation.get("asset_id")
    upload_url = reservation.get("upload_url")
    if not asset_id or reservation.get("method") != "PUT" or not upload_url:
        raise RuntimeError("Storage reservation did not return a valid PUT upload URL")

    _UPLOAD_EXECUTOR.submit(
        _upload_and_complete,
        path,
        upload_url,
        uploads_root,
        asset_id,
        file_size,
        codec,
        metadata,
        headers,
        keep_file,
        request_session,
    )
    return asset_id


def _upload_and_complete(
    path, upload_url, uploads_root, asset_id, file_size, codec,
    metadata, headers, keep_file, request_session,
):
    """Upload a reserved file and complete it in the background."""
    try:
        with path.open("rb") as image_file:
            response = request_session.put(
                upload_url,
                data=image_file,
                headers={"Content-Type": mimetypes.guess_type(path.name)[0]
                         or "application/octet-stream"},
                timeout=600,
            )
        response.raise_for_status()

        response = request_session.post(
            f"{uploads_root}/{asset_id}/complete",
            json={"file_size": file_size, "codec": codec, "extra_data": metadata},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return asset_id
    except Exception:
        logging.exception("Background MinIO upload failed for asset_id=%s", asset_id)
        raise
    finally:
        if not keep_file and path.exists():
            try:
                path.unlink()
            except OSError:
                logging.exception("Could not remove local file after upload: %s", path)
