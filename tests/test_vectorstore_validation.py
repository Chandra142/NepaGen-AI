"""Tests for vectorstore compatibility and failure handling."""
import pytest

import vectorstore_store as vs


def _manifest(**overrides: object) -> dict:
    manifest = {
        "schema_version": vs.VECTORSTORE_SCHEMA_VERSION,
        "embedding_model": vs.EMBEDDING_MODEL_NAME,
    }
    manifest.update(overrides)
    return manifest


def test_compatible_manifest_has_no_error() -> None:
    assert vs._manifest_compatibility_error(_manifest()) is None


def test_schema_mismatch_is_reported() -> None:
    error = vs._manifest_compatibility_error(_manifest(schema_version=0))
    assert error is not None and "schema" in error.lower()


def test_embedding_model_mismatch_is_reported() -> None:
    error = vs._manifest_compatibility_error(_manifest(embedding_model="other-model"))
    assert error is not None and "embedding model" in error.lower()


def test_missing_files_raise_instead_of_returning_empty_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(vs, "VECTORSTORE_DIR", tmp_path)
    monkeypatch.setattr(vs, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(vs, "DOCSTORE_PATH", tmp_path / "docstore.json")
    monkeypatch.setattr(vs, "MANIFEST_PATH", tmp_path / "manifest.json")

    with pytest.raises(vs.VectorstoreCompatibilityError, match="missing"):
        vs.load_vectorstore()
