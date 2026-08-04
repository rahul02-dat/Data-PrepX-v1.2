import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "contracts" / "job.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    with SCHEMA_PATH.open() as f:
        return json.load(f)


def _validator(schema: dict, def_name: str) -> jsonschema.Draft202012Validator:
    ref_schema = {"$ref": f"#/$defs/{def_name}", **schema}
    return jsonschema.Draft202012Validator(ref_schema)


def test_schema_itself_is_valid(schema: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"dataset_id": None},
        {"config": {"foo": "bar"}},
        {"dataset_id": "550e8400-e29b-41d4-a716-446655440000", "config": {}},
    ],
)
def test_valid_submit_requests(schema: dict, payload: dict) -> None:
    _validator(schema, "jobSubmitRequest").validate(payload)


def test_submit_request_rejects_unknown_fields(schema: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        _validator(schema, "jobSubmitRequest").validate({"unexpected_field": 1})


@pytest.mark.parametrize(
    "payload",
    [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "dataset_id": None,
            "status": "queued",
            "config_hash": "sha256:abc123",
            "created_at": "2026-08-01T23:14:08.677278Z",
            "updated_at": "2026-08-01T23:14:08.677278Z",
        },
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "done",
            "config_hash": "sha256:abc123",
            "created_at": "2026-08-01T23:14:08.677278Z",
            "updated_at": "2026-08-01T23:14:09.177278Z",
        },
    ],
)
def test_valid_job_responses(schema: dict, payload: dict) -> None:
    _validator(schema, "job").validate(payload)


def test_job_response_rejects_invalid_status(schema: dict) -> None:
    payload = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "bogus-status",
        "config_hash": "sha256:abc123",
        "created_at": "2026-08-01T23:14:08.677278Z",
        "updated_at": "2026-08-01T23:14:08.677278Z",
    }
    with pytest.raises(jsonschema.ValidationError):
        _validator(schema, "job").validate(payload)


def test_job_response_requires_config_hash(schema: dict) -> None:
    payload = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "queued",
        "created_at": "2026-08-01T23:14:08.677278Z",
        "updated_at": "2026-08-01T23:14:08.677278Z",
    }
    with pytest.raises(jsonschema.ValidationError):
        _validator(schema, "job").validate(payload)