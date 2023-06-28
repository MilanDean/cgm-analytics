import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.models import GlucoseData
from pydantic import ValidationError


client = TestClient(app)


@pytest.fixture(
    params=[
        {
            "Time": "12:00",
            "BG": 100,
            "CGM": 120,
            "CHO": 10,
            "insulin": 5,
            "LBGI": 0.5,
            "HBGI": 0.2,
            "Risk": 0.3,
        },
        {
            "Time": "08:30",
            "BG": 90,
            "CGM": 110,
            "CHO": 15,
            "insulin": 4,
            "LBGI": 0.8,
            "HBGI": 0.1,
            "Risk": 0.4,
        },
    ]
)
def valid_glucose_data(request):
    return request.param


@pytest.fixture(
    params=[
        {"Time": "12:00", "BG": 100, "CGM": 120, "CHO": 10, "insulin": 5, "LBGI": 0.5},
        {"Time": "10:15", "BG": 80, "CGM": 130},
        {"Time": "15:45", "BG": 120, "CGM": 90, "CHO": 25, "insulin": 6},
        {
            "Time": "8:00",
            "BG": 85,
            "CGM": 100,
            "CHO": 8,
            "insulin": 3,
            "LBGI": "test1",
            "HBGI": 0.2,
            "Risk": 0.1,
        },
        {
            "Time": "8:00",
            "BG": 85,
            "CGM": 100,
            "CHO": 8,
            "insulin": 3,
            "LBGI": 0.1,
            "HBGI": 0.2,
            "Risk": "test2",
        },
    ]
)
def invalid_glucose_data(request):
    return request.param


def test_valid_glucose_data(valid_glucose_data):
    glucose_data = GlucoseData(**valid_glucose_data)
    assert isinstance(glucose_data, GlucoseData)


def test_invalid_glucose_data(invalid_glucose_data):
    with pytest.raises(ValidationError):
        GlucoseData(**invalid_glucose_data)
