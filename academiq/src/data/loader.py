"""AcademIQ — Data loading and validation.

Provides DataLoader for reading the CSV and DataValidator for schema,
type, range, and categorical-value checks. Returns structured reports
instead of just printing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    get_all_input_features,
    get_nominal_features,
    get_numeric_features,
    get_ordinal_features,
    get_raw_data_path,
    get_target_column,
)

logger = logging.getLogger(__name__)


# --- Data Loader ---


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset.

    Args:
        path: Optional path override. Defaults to config-defined location.

    Returns:
        Raw DataFrame exactly as stored on disk.
    """
    if path is None:
        path = get_raw_data_path()
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


# --- Validation Report ---


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: str  # "error" or "warning"
    category: str  # e.g. "missing_column", "invalid_type", "out_of_range"
    column: str | None
    message: str


@dataclass
class ValidationReport:
    """Structured output from data validation."""

    is_valid: bool = True
    n_rows: int = 0
    n_cols: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, category: str, column: str | None, message: str):
        self.issues.append(ValidationIssue("error", category, column, message))
        self.is_valid = False

    def add_warning(self, category: str, column: str | None, message: str):
        self.issues.append(ValidationIssue("warning", category, column, message))

    def summary(self) -> str:
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        lines = [
            f"Validation: {'PASSED' if self.is_valid else 'FAILED'}",
            f"  Rows: {self.n_rows}, Columns: {self.n_cols}",
            f"  Errors: {len(errors)}, Warnings: {len(warnings)}",
        ]
        for issue in self.issues:
            col_str = f" [{issue.column}]" if issue.column else ""
            lines.append(f"  [{issue.severity.upper()}]{col_str} {issue.message}")
        return "\n".join(lines)


# --- Data Validator ---

# Expected numeric ranges (generous bounds for validation)
NUMERIC_RANGES = {
    "Hours_Studied": (0, 50),
    "Attendance": (0, 100),
    "Sleep_Hours": (0, 24),
    "Previous_Scores": (0, 120),
    "Tutoring_Sessions": (0, 50),
    "Physical_Activity": (0, 10),
    "Exam_Score": (0, 120),
}


def validate_data(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Validate a DataFrame against the expected schema.

    Args:
        df: DataFrame to validate.
        require_target: If True, check that the target column exists.

    Returns:
        ValidationReport with all issues found.
    """
    report = ValidationReport(n_rows=len(df), n_cols=len(df.columns))

    target = get_target_column()
    expected_features = get_all_input_features()
    expected_columns = expected_features + ([target] if require_target else [])

    # --- Check required columns ---
    for col in expected_columns:
        if col not in df.columns:
            report.add_error("missing_column", col, f"Required column '{col}' not found")

    # If critical columns are missing, skip further checks
    if not report.is_valid:
        return report

    # --- Check for empty DataFrame ---
    if len(df) == 0:
        report.add_error("empty_data", None, "DataFrame has 0 rows")
        return report

    # --- Check missing values ---
    for col in expected_columns:
        if col in df.columns:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                pct = n_missing / len(df) * 100
                report.add_warning(
                    "missing_values",
                    col,
                    f"{n_missing} missing values ({pct:.1f}%)",
                )

    # --- Check numeric ranges ---
    for col in get_numeric_features():
        if col not in df.columns:
            continue
        lo, hi = NUMERIC_RANGES.get(col, (None, None))
        if lo is not None:
            below = (df[col].dropna() < lo).sum()
            if below > 0:
                report.add_warning(
                    "out_of_range", col, f"{below} values below {lo}"
                )
        if hi is not None:
            above = (df[col].dropna() > hi).sum()
            if above > 0:
                report.add_warning(
                    "out_of_range", col, f"{above} values above {hi}"
                )

    # Check target range
    if require_target and target in df.columns:
        lo, hi = NUMERIC_RANGES.get(target, (None, None))
        if lo is not None:
            below = (df[target].dropna() < lo).sum()
            if below > 0:
                report.add_warning("out_of_range", target, f"{below} values below {lo}")

    # --- Check categorical values ---
    ordinal = get_ordinal_features()
    for col, allowed in ordinal.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        unexpected = actual - set(allowed)
        if unexpected:
            report.add_error(
                "invalid_category",
                col,
                f"Unexpected values: {unexpected}. Allowed: {allowed}",
            )

    nominal = get_nominal_features()
    # Define allowed values for nominal features
    nominal_allowed = {
        "Gender": ["Male", "Female"],
        "Extracurricular_Activities": ["Yes", "No"],
        "Internet_Access": ["Yes", "No"],
        "School_Type": ["Public", "Private"],
        "Learning_Disabilities": ["Yes", "No"],
    }
    for col in nominal:
        if col not in df.columns:
            continue
        allowed = nominal_allowed.get(col)
        if allowed is None:
            continue
        actual = set(df[col].dropna().unique())
        unexpected = actual - set(allowed)
        if unexpected:
            report.add_error(
                "invalid_category",
                col,
                f"Unexpected values: {unexpected}. Allowed: {allowed}",
            )

    # --- Check duplicates ---
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        report.add_warning("duplicates", None, f"{n_dupes} duplicate rows found")

    logger.info("Validation complete: %s", "PASSED" if report.is_valid else "FAILED")
    return report
