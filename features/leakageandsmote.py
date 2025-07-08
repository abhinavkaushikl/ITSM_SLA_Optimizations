import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from collections import Counter


class LeakyFeatureRemover(BaseEstimator, TransformerMixin):
    """
    Removes leaky columns based on keywords and separates features and target.
    """
    def __init__(self, target_col='SLA Breach', leakage_keywords=None, verbose=True):
        self.target_col = target_col
        self.verbose = verbose
        self.leakage_keywords = leakage_keywords or [
            'resolved', 'resolution', 'response', 'sla', 'csat', 'penalty', 'mttr', 'mtbf'
        ]
        self.leaky_columns_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_col not in X.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the dataframe.")
        self.leaky_columns_ = [
            col for col in X.columns if any(k in col.lower() for k in self.leakage_keywords)
        ]
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.verbose:
            print(f" Dropping {len(self.leaky_columns_)} leaky columns: {self.leaky_columns_}")
        X_dropped = X.drop(columns=self.leaky_columns_ + [self.target_col])
        y = X[self.target_col]
        return X_dropped, y


class SMOTEHandler:
    """
    Applies SMOTE to balance imbalanced target classes.
    """
    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.sampler = SMOTE(random_state=self.random_state)

    def apply(self, X: pd.DataFrame, y: pd.Series):
        if self.verbose:
            print(f"Class distribution before SMOTE: {Counter(y)}")
        X_res, y_res = self.sampler.fit_resample(X, y)
        if self.verbose:
            print(f"Class distribution after SMOTE:  {Counter(y_res)}")
        return X_res, y_res


def main():
    # Step 1: Load the encoded DataFrame
    try:
        df = pd.read_csv("data/processed/encoded_data.csv")  # Update path if needed
        print(" Loaded encoded data successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Step 2: Remove leaky features
    remover = LeakyFeatureRemover(target_col='SLA Breach', verbose=True)
    X, y = remover.fit_transform(df)

    print(f"Cleaned features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Step 3: Balance with SMOTE
    smoter = SMOTEHandler(verbose=True)
    X_resampled, y_resampled = smoter.apply(X, y)

    print(f"Final balanced dataset shape: {X_resampled.shape}, {y_resampled.shape}")


if __name__ == "__main__":
    main()