import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype, is_bool_dtype
import warnings
warnings.filterwarnings('ignore')

class FeatureEncoder:
    def __init__(self, df, target_col='SLA Breach', verbose=True):
        self.df = df.copy()
        self.target_col = target_col
        self.verbose = verbose
        self.encoders = {}
        self.strategy_df = pd.DataFrame()

    def _is_engineered_feature(self, col):
        keywords = ['hour', 'day', 'month', 'year', 'time']
        return any(k in col.lower() for k in keywords) and is_numeric_dtype(self.df[col])

    def encode(self):
        strategies = []
        df = self.df

        for col in df.columns:
            if col == self.target_col:
                continue

            n_unique = df[col].nunique()
            col_type = df[col].dtype

            if self._is_engineered_feature(col):
                strategy = 'Already Feature Engineered'

            elif is_numeric_dtype(df[col]):
                strategy = 'Numeric - No Encoding'

            elif is_bool_dtype(df[col]):
                strategy = 'Boolean - Binary Encoding'
                df[col] = df[col].astype(int)

            elif n_unique <= 10:
                strategy = 'Categorical - OneHotEncoding'
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

            elif n_unique > 10:
                strategy = 'Categorical - LabelEncoding'
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le

            else:
                strategy = 'Unknown'

            strategies.append({
                'Column': col,
                'Dtype': str(col_type),
                'Unique Values': n_unique,
                'Encoding Strategy': strategy
            })

        self.df = df
        self.strategy_df = pd.DataFrame(strategies).sort_values(by='Encoding Strategy')

        if self.verbose:
            print("\n Encoding Strategy Overview:")
            print(self.strategy_df.to_string(index=False))

        return self.df, self.strategy_df, self.encoders

# if __name__ == "__main__":
   
#    test_data =  pd.read_csv("data/processed/processed_time_series_data.csv")
#    df_test = pd.DataFrame(test_data)
#    print("Original DataFrame:\n", df_test)

#    encoder = FeatureEncoder(df_test, target_col='SLA Breach', verbose=True)
#    df_encoded, encoding_strategy_df, encoders = encoder.encode()
    
#    df_encoded.to_csv("data/processed/encoded_data.csv", index=False)
#    print("\n Encoded DataFrame:\n", df_encoded.head())