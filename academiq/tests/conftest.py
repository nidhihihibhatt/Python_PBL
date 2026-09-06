"""Shared test fixtures for AcademIQ test suite."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_student():
    """A single valid student as a dict."""
    return {
        "Hours_Studied": 20,
        "Attendance": 80,
        "Parental_Involvement": "Medium",
        "Access_to_Resources": "High",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours": 7,
        "Previous_Scores": 72,
        "Motivation_Level": "High",
        "Internet_Access": "Yes",
        "Tutoring_Sessions": 2,
        "Family_Income": "Medium",
        "Teacher_Quality": "Medium",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Physical_Activity": 3,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
        "Gender": "Female",
    }


@pytest.fixture
def sample_dataframe(sample_student):
    """A 5-row DataFrame with valid students."""
    students = []
    for i in range(5):
        s = sample_student.copy()
        s["Hours_Studied"] = 10 + i * 5
        s["Attendance"] = 60 + i * 8
        s["Previous_Scores"] = 60 + i * 5
        students.append(s)
    return pd.DataFrame(students)


@pytest.fixture
def sample_dataframe_with_target(sample_dataframe):
    """DataFrame with Exam_Score target column."""
    df = sample_dataframe.copy()
    df["Exam_Score"] = [62, 65, 68, 70, 73]
    return df


@pytest.fixture
def sample_shap_values():
    """Synthetic SHAP values for recommendation testing."""
    return {
        "Hours_Studied": -0.5,
        "Attendance": -1.2,
        "Previous_Scores": -0.3,
        "Tutoring_Sessions": -0.1,
        "Parental_Involvement": -0.4,
        "Access_to_Resources": -0.2,
        "Motivation_Level": -0.15,
        "Sleep_Hours": 0.1,
        "Physical_Activity": 0.05,
    }
