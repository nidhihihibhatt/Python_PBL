"""AcademIQ — Custom sklearn transformers.

Contains FeatureEngineer, a proper sklearn TransformerMixin that creates
engineered features inside a Pipeline. This ensures the same transformations
are applied at training and inference time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create engineered features from raw (already-encoded) input.

    Designed to sit inside an sklearn Pipeline AFTER the ColumnTransformer
    has converted categoricals to numeric. The transformer expects the
    following columns to exist in the input array (by index, set during fit):

    - Hours_Studied (numeric)
    - Attendance (numeric)
    - Parental_Involvement (ordinal-encoded: 0, 1, 2)
    - Access_to_Resources (ordinal-encoded: 0, 1, 2)
    - Tutoring_Sessions (numeric)
    - Motivation_Level (ordinal-encoded: 0, 1, 2)
    - Extracurricular_Activities (one-hot encoded)

    Features created:
    - Study_Effort = Hours_Studied × Attendance / 100
    - Support_Index = (Parental_Involvement + Access_to_Resources + norm(Tutoring_Sessions)) / 3
    - Engagement = Motivation_Level + Extracurricular_Activities

    Parameters
    ----------
    add_study_effort : bool, default=True
        Whether to add the Study_Effort feature.
    add_support_index : bool, default=True
        Whether to add the Support_Index feature.
    add_engagement : bool, default=True
        Whether to add the Engagement feature.
    """

    def __init__(
        self,
        add_study_effort: bool = True,
        add_support_index: bool = True,
        add_engagement: bool = True,
    ):
        self.add_study_effort = add_study_effort
        self.add_support_index = add_support_index
        self.add_engagement = add_engagement

    def fit(self, X, y=None):
        """Learn column positions from the input feature names.

        Works with both DataFrames and numpy arrays (via Pipeline feature names).
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        elif hasattr(self, "_feature_names_in"):
            # Set by pipeline via set_output or get_feature_names_out
            pass
        self.n_features_in_ = X.shape[1]

        # Store the column indices for the features we need.
        # These names come from the ColumnTransformer output.
        self._col_map = {}
        names = self._get_feature_names(X)
        if names is not None:
            for i, name in enumerate(names):
                self._col_map[name] = i
        return self

    def transform(self, X):
        """Add engineered features as new columns."""
        if isinstance(X, pd.DataFrame):
            result = X.copy()
        else:
            # Convert to DataFrame for easier manipulation
            names = self._get_feature_names(X)
            if names is not None:
                result = pd.DataFrame(X, columns=names)
            else:
                result = pd.DataFrame(X)

        new_features = {}

        if self.add_study_effort:
            hours = self._get_col(result, "Hours_Studied")
            attendance = self._get_col(result, "Attendance")
            if hours is not None and attendance is not None:
                new_features["Study_Effort"] = hours * attendance / 100.0

        if self.add_support_index:
            parental = self._get_col(result, "Parental_Involvement")
            resources = self._get_col(result, "Access_to_Resources")
            tutoring = self._get_col(result, "Tutoring_Sessions")
            if parental is not None and resources is not None and tutoring is not None:
                # Normalise tutoring to [0, 2] range to match ordinal scale
                tutoring_max = self._tutoring_max if hasattr(self, "_tutoring_max") else 8
                tutoring_norm = np.clip(tutoring / max(tutoring_max, 1) * 2, 0, 2)
                new_features["Support_Index"] = (
                    parental + resources + tutoring_norm
                ) / 3.0

        if self.add_engagement:
            motivation = self._get_col(result, "Motivation_Level")
            # Extracurricular may be one-hot encoded with various names
            extra = self._get_col(result, "Extracurricular_Activities")
            if extra is None:
                extra = self._get_col(result, "Extracurricular_Activities_Yes")
            if motivation is not None and extra is not None:
                new_features["Engagement"] = motivation + extra

        for name, values in new_features.items():
            result[name] = values

        return result

    def fit_transform(self, X, y=None):
        """Fit and transform, also learning tutoring max for normalisation."""
        self.fit(X, y)
        # Learn tutoring max from training data
        tutoring = self._get_col_from_raw(X, "Tutoring_Sessions")
        if tutoring is not None:
            self._tutoring_max = float(np.max(tutoring))
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return output feature names including engineered features."""
        if input_features is not None:
            names = list(input_features)
        elif hasattr(self, "feature_names_in_"):
            names = list(self.feature_names_in_)
        else:
            names = [f"x{i}" for i in range(self.n_features_in_)]

        if self.add_study_effort:
            names.append("Study_Effort")
        if self.add_support_index:
            names.append("Support_Index")
        if self.add_engagement:
            names.append("Engagement")
        return names

    # --- Internal helpers ---

    def _get_feature_names(self, X):
        """Get feature names from X if available."""
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return None

    def _get_col(self, df: pd.DataFrame, name: str):
        """Get a column by name, returning None if not found."""
        if name in df.columns:
            return df[name].values.astype(float)
        return None

    def _get_col_from_raw(self, X, name: str):
        """Get column from raw X (DataFrame or array)."""
        if isinstance(X, pd.DataFrame) and name in X.columns:
            return X[name].values
        names = self._get_feature_names(X)
        if names and name in names:
            idx = names.index(name)
            return X[:, idx] if hasattr(X, "__getitem__") else None
        return None
