import json
import io
from uuid import uuid4
import streamlit as st
import pandas as pd

from firebase_service import (
    FirebaseError,
    build_file_record,
    build_message_record,
    delete_document,
    download_storage_file,
    get_document,
    list_documents,
    set_document,
    upload_storage_file,
)

def encrypt_credentials(credentials_dict: dict) -> str:
    """Retorna o dicionário de credenciais como string JSON (sem criptografia)."""
    return json.dumps(credentials_dict)

def decrypt_credentials(encrypted_string: str) -> dict:
    """Converte a string JSON de volta para um dicionário."""
    if not encrypted_string: return None
    try:
        return json.loads(encrypted_string)
    except:
        return None

# --- Configurações Locais ---

def _get_active_user():
    user = st.session_state.get("user")
    if not user or not user.get("user_id") or not user.get("idToken"):
        raise FirebaseError("Usuario autenticado nao encontrado na sessao.")
    return user


def _get_user_id(user_id: str | None = None) -> str:
    return user_id or _get_active_user()["user_id"]


def _get_id_token() -> str:
    return _get_active_user()["idToken"]


def _sort_by_timestamp(items: list[dict], key: str) -> list[dict]:
    return sorted(items, key=lambda item: item.get(key, ""))


def restore_user_session(user: dict):
    """Hidrata o estado local do Streamlit com os dados persistidos do usuario."""
    st.session_state["user"] = user
    st.session_state["logged_in"] = True
    st.session_state["history"] = fetch_chat_history(user["user_id"])
    st.session_state["charts"] = []
    st.session_state["db_creds"] = get_user_config(user["user_id"]) or {}
    st.session_state["user_files"] = list_user_files(user["user_id"])

    if st.session_state["user_files"]:
        latest_file = _sort_by_timestamp(st.session_state["user_files"], "uploaded_at")[-1]
        st.session_state["selected_file_id"] = latest_file["file_id"]
    else:
        st.session_state["selected_file_id"] = None

    st.session_state.pop("df_data", None)
    st.session_state.pop("loaded_file_id", None)
    st.session_state.pop("connected_uri", None)
    st.session_state.pop("schema_text", None)
    st.session_state.pop("schema_dict", None)


def get_user_config(user_id: str):
    """Busca as credenciais salvas do banco de dados no Firestore."""
    document = get_document(f"users/{user_id}/settings/db_config", _get_id_token())
    if not document:
        return None

    if "credentials_json" in document:
        return decrypt_credentials(document["credentials_json"])

    return document.get("credentials")

def save_db_config(user_id: str, credentials_dict: dict):
    """Salva as credenciais do banco de dados do usuario no Firestore."""
    set_document(
        f"users/{user_id}/settings/db_config",
        {
            "credentials_json": encrypt_credentials(credentials_dict),
            "updated_at": build_message_record("system", "db_config")["created_at"],
        },
        _get_id_token(),
    )
    return True


def fetch_chat_history(user_id: str | None = None) -> list[dict]:
    """Carrega o historico do usuario a partir do Firestore."""
    uid = _get_user_id(user_id)
    documents = list_documents(f"users/{uid}/history", _get_id_token())
    history = [
        {
            "message_id": doc.get("message_id", ""),
            "role": doc.get("role", ""),
            "content": doc.get("content", ""),
            "created_at": doc.get("created_at", ""),
        }
        for doc in documents
    ]
    return _sort_by_timestamp(history, "created_at")


def append_chat_message(role: str, content: str) -> dict:
    """Persiste uma mensagem de chat no Firestore e atualiza o estado local."""
    user = _get_active_user()
    message = build_message_record(role, content)
    set_document(
        f"users/{user['user_id']}/history/{message['message_id']}",
        message,
        user["idToken"],
    )
    st.session_state.setdefault("history", []).append(
        {"message_id": message["message_id"], "role": role, "content": content}
    )
    return message


def clear_chat_history() -> None:
    """Remove o historico do usuario no Firestore e localmente."""
    user = _get_active_user()
    documents = list_documents(f"users/{user['user_id']}/history", user["idToken"])
    for document in documents:
        message_id = document.get("message_id")
        if not message_id:
            continue
        delete_document(f"users/{user['user_id']}/history/{message_id}", user["idToken"])
    st.session_state["history"] = []


def list_user_files(user_id: str | None = None) -> list[dict]:
    """Lista os arquivos persistidos do usuario."""
    uid = _get_user_id(user_id)
    documents = list_documents(f"users/{uid}/files", _get_id_token())
    files = [
        {
            "file_id": doc.get("file_id", ""),
            "name": doc.get("name", ""),
            "storage_path": doc.get("storage_path", ""),
            "content_type": doc.get("content_type", ""),
            "size_bytes": doc.get("size_bytes", 0),
            "uploaded_at": doc.get("uploaded_at", ""),
        }
        for doc in documents
        if doc.get("file_id")
    ]
    return _sort_by_timestamp(files, "uploaded_at")


def save_uploaded_file(uploaded_file) -> dict:
    """Faz upload para o Firebase Storage e persiste os metadados no Firestore."""
    user = _get_active_user()
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    file_size = len(file_bytes)
    storage_path = f"users/{user['user_id']}/files/{uuid4().hex}/{uploaded_file.name}"

    storage_payload = upload_storage_file(
        file_name=file_name,
        file_bytes=file_bytes,
        id_token=user["idToken"],
        storage_path=storage_path,
        content_type=getattr(uploaded_file, "type", None),
    )
    file_record = build_file_record(file_name, storage_payload, file_size)
    set_document(
        f"users/{user['user_id']}/files/{file_record['file_id']}",
        file_record,
        user["idToken"],
    )
    current_files = [file for file in st.session_state.get("user_files", []) if file["file_id"] != file_record["file_id"]]
    current_files.append(file_record)
    st.session_state["user_files"] = _sort_by_timestamp(current_files, "uploaded_at")
    st.session_state["selected_file_id"] = file_record["file_id"]
    return file_record


def load_dataframe_from_storage(file_record: dict) -> pd.DataFrame:
    """Baixa um arquivo do Storage e o carrega em um DataFrame."""
    file_bytes = download_storage_file(file_record["storage_path"], _get_id_token())
    file_name = file_record["name"].lower()
    buffer = io.BytesIO(file_bytes)
    if file_name.endswith(".csv"):
        return pd.read_csv(buffer)
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(buffer)
    raise ValueError("Formato de arquivo nao suportado. Use CSV ou Excel.")
