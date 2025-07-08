import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesProcessor:
    def __init__(self, df, datetime_columns=None, reference_date_col=None):
        self.df = df.copy()
        self.datetime_columns = datetime_columns or self._auto_detect_datetime_columns()
        self.reference_date_col = reference_date_col

    def _auto_detect_datetime_columns(self):
        detected = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col].head(100), errors='raise')
                    detected.append(col)
                except:
                    continue
        print("Auto-detected datetime columns:", detected)
        return detected

    def _categorize_time_of_day(self, hour):
        if 6 <= hour < 12:
            return 1  # Morning
        elif 12 <= hour < 18:
            return 2  # Afternoon
        elif 18 <= hour < 22:
            return 3  # Evening
        else:
            return 4  # Night

    def _extract_temporal_features(self, datetime_col):
        print(f"  Extracting temporal features from {datetime_col}...")
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col], errors='coerce')

        df = self.df
        df[f'{datetime_col}_year'] = df[datetime_col].dt.year
        df[f'{datetime_col}_month'] = df[datetime_col].dt.month
        df[f'{datetime_col}_day'] = df[datetime_col].dt.day
        df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
        df[f'{datetime_col}_minute'] = df[datetime_col].dt.minute
        df[f'{datetime_col}_dayofweek'] = df[datetime_col].dt.dayofweek
        df[f'{datetime_col}_dayofyear'] = df[datetime_col].dt.dayofyear
        df[f'{datetime_col}_week'] = df[datetime_col].dt.isocalendar().week
        df[f'{datetime_col}_quarter'] = df[datetime_col].dt.quarter

        df[f'{datetime_col}_is_weekend'] = (df[datetime_col].dt.dayofweek >= 5).astype(int)
        df[f'{datetime_col}_is_monday'] = (df[datetime_col].dt.dayofweek == 0).astype(int)
        df[f'{datetime_col}_is_friday'] = (df[datetime_col].dt.dayofweek == 4).astype(int)

        df[f'{datetime_col}_time_category'] = df[datetime_col].dt.hour.apply(self._categorize_time_of_day)

        df[f'{datetime_col}_is_business_hours'] = (
            (df[datetime_col].dt.hour >= 9) & (df[datetime_col].dt.hour < 17) & (df[datetime_col].dt.dayofweek < 5)
        ).astype(int)

        df[f'{datetime_col}_is_month_end'] = (
            df[datetime_col].dt.day >= df[datetime_col].dt.days_in_month - 2
        ).astype(int)

        df[f'{datetime_col}_hour_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.hour / 24)
        df[f'{datetime_col}_hour_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.hour / 24)
        df[f'{datetime_col}_dayofweek_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.dayofweek / 7)
        df[f'{datetime_col}_dayofweek_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.dayofweek / 7)
        df[f'{datetime_col}_month_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.month / 12)
        df[f'{datetime_col}_month_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.month / 12)

    def _calculate_itsm_metrics(self):
        print("\nCalculating ITSM-specific time metrics...")

        mapping = {
            'created_date': 'creation', 'resolved_date': 'resolution', 'closed_date': 'closure',
            'first_response_date': 'first_response', 'last_updated_date': 'last_update',
            'due_date': 'due', 'escalated_date': 'escalation'
        }

        available = {
            label: col for col in self.datetime_columns
            for key, label in mapping.items() if key in col.lower() or label in col.lower()
        }

        df = self.df

        if 'creation' in available and 'resolution' in available:
            df['resolution_time_hours'] = (
                df[available['resolution']] - df[available['creation']]
            ).dt.total_seconds() / 3600
            df['resolution_time_days'] = df['resolution_time_hours'] / 24

        if 'creation' in available and 'first_response' in available:
            df['first_response_time_hours'] = (
                df[available['first_response']] - df[available['creation']]
            ).dt.total_seconds() / 3600

        if 'creation' in available and 'due' in available:
            df['time_to_due_hours'] = (
                df[available['due']] - df[available['creation']]
            ).dt.total_seconds() / 3600

            if 'resolution' in available:
                df['sla_breached'] = (
                    df[available['resolution']] > df[available['due']]
                ).astype(int)
                df['sla_breach_hours'] = np.where(
                    df['sla_breached'] == 1,
                    (df[available['resolution']] - df[available['due']]).dt.total_seconds() / 3600,
                    0
                )

        if self.reference_date_col and self.reference_date_col in available:
            now = datetime.now()
            df['ticket_age_hours'] = (
                now - df[available[self.reference_date_col]]
            ).dt.total_seconds() / 3600
            df['ticket_age_days'] = df['ticket_age_hours'] / 24

    def _calculate_business_features(self):
        print("\nCalculating business time features...")
        df = self.df
        year_start = pd.Timestamp(f'{datetime.now().year}-01-01')

        for col in self.datetime_columns:
            df[f'{col}_business_days_from_year_start'] = df[col].apply(
                lambda x: np.busday_count(year_start.date(), x.date()) if pd.notna(x) else np.nan
            )
            df[f'{col}_is_peak_hours'] = df[col].dt.hour.isin([9, 10, 11, 14, 15, 16]).astype(int)

    def process(self):
        print("=" * 60)
        print("TIME SERIES DATA PROCESSING FOR ITSM SLA OPTIMIZATION")
        print("=" * 60)

        for col in self.datetime_columns:
            self._extract_temporal_features(col)

        self._calculate_itsm_metrics()
        self._calculate_business_features()

        print("\n✓ Time series data processing completed!")
        print(f"✓ Final dataset shape: {self.df.shape}")
        return self.df

    def summarize_new_features(self, original_columns):
        print("\n=" * 25)
        print("TIME SERIES PROCESSING RESULTS")
        print("=" * 50)

        new_cols = set(self.df.columns) - set(original_columns)
        print(f"New features created ({len(new_cols)}):")
        for col in sorted(new_cols):
            print(f"  - {col}")

        print("\nSample of new features:")
        time_cols = [col for col in self.df.columns if any(k in col.lower() for k in ['time', 'hour', 'sla', 'age'])]
        print(self.df[time_cols[:10]].head())

        print("\nSummary of time metrics:")
        key_metrics = [col for col in self.df.columns if any(k in col.lower() for k in ['resolution_time', 'response_time', 'sla_breach', 'ticket_age'])]



# if __name__ == "__main__":
#     import sys


#     csv_path = "data/processed/processed_null_values_data.csv"

#     try:
#        df = pd.read_csv(csv_path)
#        print(f"\n Loaded dataset with shape: {df.shape}")
#     except FileNotFoundError:
#        print(f" File not found at: {csv_path}")
#        sys.exit(1)


#     original_cols = df.columns.tolist()


#     processor = TimeSeriesProcessor(df, reference_date_col='created_date')
#     df_processed = processor.process()
#     processor.summarize_new_features(original_columns=original_cols)

#     output_path = "data/processed/processed_time_series_data.csv"   
#     df_processed.to_csv(output_path, index=False)
#     print(f"\nProcessed data saved to: {output_path}")