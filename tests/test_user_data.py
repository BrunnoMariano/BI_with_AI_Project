"""Testes da criptografia da chave OpenAI padrão (sem Firebase)."""

from cryptography.fernet import Fernet

from user_data import decrypt_secret, encrypt_secret


def test_chave_criptografada_volta_igual():
    chave_app = Fernet.generate_key().decode()
    token = encrypt_secret("sk-teste-123", chave_app)
    assert "sk-teste-123" not in token
    assert decrypt_secret(token, chave_app) == "sk-teste-123"


def test_chave_de_criptografia_errada_retorna_none():
    token = encrypt_secret("sk-teste-123", Fernet.generate_key().decode())
    assert decrypt_secret(token, Fernet.generate_key().decode()) is None
