import io
import mimetypes
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import requests

from config import get_firebase_config


class FirebaseError(Exception):
    """Erro padrao de integracao com o Firebase."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _firebase_config() -> dict[str, str]:
    return get_firebase_config()


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error", {})
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return message
        if isinstance(error, str):
            return error
    return str(payload)


def _request_json(method: str, url: str, **kwargs) -> dict[str, Any]:
    response = requests.request(method, url, timeout=45, **kwargs)
    if not response.ok:
        raise FirebaseError(_extract_error_message(response))
    if not response.content:
        return {}
    return response.json()


def _normalize_auth_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": payload.get("localId", ""),
        "email": payload.get("email", ""),
        "idToken": payload.get("idToken", ""),
        "refreshToken": payload.get("refreshToken", ""),
        "expiresIn": payload.get("expiresIn", ""),
    }


def sign_in_user(email: str, password: str) -> dict[str, Any]:
    config = _firebase_config()
    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        f"?key={config['apiKey']}"
    )
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }
    return _normalize_auth_payload(_request_json("POST", url, json=payload))


def register_user_account(email: str, password: str) -> dict[str, Any]:
    config = _firebase_config()
    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
        f"?key={config['apiKey']}"
    )
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }
    return _normalize_auth_payload(_request_json("POST", url, json=payload))


def lookup_user_account(id_token: str) -> dict[str, Any]:
    config = _firebase_config()
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={config['apiKey']}"
    payload = {"idToken": id_token}
    data = _request_json("POST", url, json=payload)
    users = data.get("users", [])
    if not users:
        raise FirebaseError("Nao foi possivel validar o usuario atual.")
    account = users[0]
    return {
        "user_id": account.get("localId", ""),
        "email": account.get("email", ""),
        "emailVerified": account.get("emailVerified", False),
    }


def send_email_verification(id_token: str) -> dict[str, Any]:
    config = _firebase_config()
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={config['apiKey']}"
    payload = {
        "requestType": "VERIFY_EMAIL",
        "idToken": id_token,
    }
    return _request_json("POST", url, json=payload)


def _firestore_headers(id_token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if id_token:
        headers["Authorization"] = f"Bearer {id_token}"
    return headers


def _firestore_document_url(document_path: str) -> str:
    project_id = _firebase_config()["projectId"]
    return (
        "https://firestore.googleapis.com/v1/projects/"
        f"{project_id}/databases/(default)/documents/{document_path}"
    )


def _encode_firestore_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, dict):
        return {
            "mapValue": {
                "fields": {key: _encode_firestore_value(item) for key, item in value.items()}
            }
        }
    if isinstance(value, list):
        return {"arrayValue": {"values": [_encode_firestore_value(item) for item in value]}}
    return {"stringValue": str(value)}


def _decode_firestore_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "booleanValue" in value:
        return value["booleanValue"]
    if "nullValue" in value:
        return None
    if "timestampValue" in value:
        return value["timestampValue"]
    if "mapValue" in value:
        fields = value["mapValue"].get("fields", {})
        return {key: _decode_firestore_value(item) for key, item in fields.items()}
    if "arrayValue" in value:
        return [_decode_firestore_value(item) for item in value["arrayValue"].get("values", [])]
    return value


def _decode_firestore_document(document: dict[str, Any]) -> dict[str, Any]:
    fields = document.get("fields", {})
    decoded = {key: _decode_firestore_value(value) for key, value in fields.items()}
    decoded["_document_name"] = document.get("name", "")
    return decoded


def set_document(document_path: str, data: dict[str, Any], id_token: str | None = None) -> dict[str, Any]:
    url = _firestore_document_url(document_path)
    payload = {"fields": {key: _encode_firestore_value(value) for key, value in data.items()}}
    response = _request_json(
        "PATCH",
        url,
        json=payload,
        headers=_firestore_headers(id_token),
    )
    return _decode_firestore_document(response)


def get_document(document_path: str, id_token: str | None = None) -> dict[str, Any] | None:
    url = _firestore_document_url(document_path)
    response = requests.get(url, headers=_firestore_headers(id_token), timeout=45)
    if response.status_code == 404:
        return None
    if not response.ok:
        raise FirebaseError(_extract_error_message(response))
    return _decode_firestore_document(response.json())


def list_documents(collection_path: str, id_token: str | None = None) -> list[dict[str, Any]]:
    url = _firestore_document_url(collection_path)
    response = requests.get(url, headers=_firestore_headers(id_token), timeout=45)
    if response.status_code == 404:
        return []
    if not response.ok:
        raise FirebaseError(_extract_error_message(response))
    payload = response.json()
    documents = payload.get("documents", [])
    return [_decode_firestore_document(document) for document in documents]


def delete_document(document_path: str, id_token: str | None = None) -> None:
    url = _firestore_document_url(document_path)
    response = requests.delete(url, headers=_firestore_headers(id_token), timeout=45)
    if response.status_code == 404:
        return
    if not response.ok:
        raise FirebaseError(_extract_error_message(response))


def upload_storage_file(
    file_name: str,
    file_bytes: bytes,
    id_token: str,
    storage_path: str | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    config = _firebase_config()
    bucket = config["storageBucket"]
    final_path = storage_path or f"users/uploads/{uuid4().hex}/{file_name}"
    detected_type = content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    url = f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o"
    params = {"uploadType": "media", "name": final_path}
    headers = {
        "Content-Type": detected_type,
        "Authorization": f"Firebase {id_token}",
    }

    response = requests.post(url, params=params, headers=headers, data=file_bytes, timeout=90)
    if response.status_code == 401:
        headers["Authorization"] = f"Bearer {id_token}"
        response = requests.post(url, params=params, headers=headers, data=file_bytes, timeout=90)
    if not response.ok:
        raise FirebaseError(_extract_error_message(response))

    payload = response.json()
    return {
        "name": payload.get("name", final_path),
        "bucket": payload.get("bucket", bucket),
        "contentType": payload.get("contentType", detected_type),
        "size": int(payload.get("size", len(file_bytes))),
        "downloadTokens": payload.get("downloadTokens", ""),
    }


def download_storage_file(storage_path: str, id_token: str) -> bytes:
    bucket = _firebase_config()["storageBucket"]
    encoded_path = quote(storage_path, safe="")
    url = f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o/{encoded_path}"
    headers = {"Authorization": f"Firebase {id_token}"}
    response = requests.get(url, params={"alt": "media"}, headers=headers, timeout=90)
    if response.status_code == 401:
        headers["Authorization"] = f"Bearer {id_token}"
        response = requests.get(url, params={"alt": "media"}, headers=headers, timeout=90)
    if not response.ok:
        raise FirebaseError(_extract_error_message(response))
    return response.content


def build_message_record(role: str, content: str) -> dict[str, Any]:
    return {
        "message_id": uuid4().hex,
        "role": role,
        "content": content,
        "created_at": _utc_now_iso(),
    }


def build_file_record(file_name: str, storage_payload: dict[str, Any], size_bytes: int) -> dict[str, Any]:
    return {
        "file_id": uuid4().hex,
        "name": file_name,
        "storage_path": storage_payload["name"],
        "bucket": storage_payload["bucket"],
        "content_type": storage_payload["contentType"],
        "size_bytes": size_bytes,
        "uploaded_at": _utc_now_iso(),
    }
