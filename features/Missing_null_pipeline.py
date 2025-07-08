import logging

import pandas as pd
import mlflow
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.df_cleaned = df.copy()

    def remove_high_null_columns(self, threshold=70):
        null_percentage = self.df_cleaned.isnull().mean() * 100
        cols_to_remove = null_percentage[null_percentage > threshold].index.tolist()
        if cols_to_remove:
            self.df_cleaned.drop(columns=cols_to_remove, inplace=True)
            logger.info(f"Removed columns with >{threshold}% nulls: {cols_to_remove}")

    def handle_missing_values(self):
        logger.info("Handling missing values...")

        # 1. Handle 'Escalation Level'
        if 'Escalation Level' in self.df_cleaned.columns:
            self.df_cleaned['Escalation Level'] = self.df_cleaned['Escalation Level'].fillna('Unknown')
            logger.info("Filled missing values in 'Escalation Level' with 'Unknown'.")

        # 2. Other categorical columns
        categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns.tolist()
        if 'Escalation Level' in categorical_cols:
            categorical_cols.remove('Escalation Level')  # Already handled

        for col in categorical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mode_val = self.df_cleaned[col].mode()
                if not mode_val.empty:
                    self.df_cleaned[col] = self.df_cleaned[col].fillna(mode_val[0])
                    logger.info(f"Filled missing in categorical column '{col}' with mode: {mode_val[0]}")
                else:
                    logger.warning(f"Skipped '{col}' (no mode found).")

        # 3. Numeric columns
        numeric_cols = self.df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                median_val = self.df_cleaned[col].median()
                self.df_cleaned[col] = self.df_cleaned[col].fillna(median_val)
                logger.info(f"Filled missing in numeric column '{col}' with median: {median_val}")

        # 4. Final missing check
        remaining = self.df_cleaned.isnull().sum()
        remaining = remaining[remaining > 0]

        if len(remaining) == 0:
            logger.info(" All missing values handled successfully!")
        else:
            logger.warning(" Missing values remain in columns:")
            for col, count in remaining.items():
                logger.warning(f"    {col}: {count} missing")

    def find_columns_with_nulls(self):
        nulls = self.df_cleaned.isnull().mean() * 100
        return nulls[nulls > 0].index.tolist()


# if __name__ == "__main__":
#     df = pd.read_csv("data/raw/itsm_sla_tickets_dataset_extended.csv")
#     processor = DataProcessor(df)
    
#     processor.remove_high_null_columns()
#     processor.handle_missing_values()
    
    

#     logger.info("Final Cleaned DataFrame Preview:")
#     logger.info(processor.df_cleaned.head())
    
    
#     processor.df_cleaned.to_csv("data/processed/processed_null_values_data.csv", index=False)

#     total_null_values = processor.df_cleaned.isnull().sum().sum()
#     logger.info(f" Total remaining null values in cleaned data: {total_null_values}")
    
