"""Tests for API key authentication."""

from __future__ import annotations

import pytest

from api.auth import APIKeyStore


class TestAPIKeyStore:
    def test_create_and_verify_key(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        key = store.create_key("test_label")
        record = store.verify_key(key)
        assert record.label == "test_label"

    def test_invalid_key_raises(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        with pytest.raises(ValueError, match="Invalid API key"):
            store.verify_key("not_a_real_key")

    def test_revoke_key(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        key = store.create_key("to_revoke")
        assert store.revoke_key(key)
        with pytest.raises(ValueError, match="revoked"):
            store.verify_key(key)

    def test_revoke_nonexistent_returns_false(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        assert not store.revoke_key("nonexistent_key")

    def test_scope_check_passes(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        key = store.create_key("scoped", scopes=["transcribe"])
        record = store.verify_key(key, required_scope="transcribe")
        assert record.label == "scoped"

    def test_scope_check_fails(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        key = store.create_key("limited", scopes=["health"])
        with pytest.raises(ValueError, match="scope"):
            store.verify_key(key, required_scope="transcribe")

    def test_request_count_increments(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        key = store.create_key("counter")
        store.verify_key(key)
        store.verify_key(key)
        record = store.verify_key(key)
        assert record.request_count == 3

    def test_list_keys(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        store.create_key("k1")
        store.create_key("k2")
        keys = store.list_keys()
        assert len(keys) == 2
        labels = {k["label"] for k in keys}
        assert "k1" in labels
        assert "k2" in labels

    def test_key_format(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        key = store.create_key("fmt_test")
        assert key.startswith("wsk_")

    def test_two_keys_independent(self) -> None:
        store = APIKeyStore(master_secret="testsecret")
        k1 = store.create_key("a")
        k2 = store.create_key("b")
        r1 = store.verify_key(k1)
        r2 = store.verify_key(k2)
        assert r1.label == "a"
        assert r2.label == "b"
# APIKeyStore._keys dict access is not thread-safe; noted as known limitation
