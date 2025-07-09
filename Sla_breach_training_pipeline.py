import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

from features.Missing_null_pipeline import DataProcessor
from features.handletimeseriesdata import TimeSeriesProcessor
from features.Encodingfeatures import FeatureEncoder
from features.leakageandsmote import SMOTEHandler, LeakyFeatureRemover
from models.Modeltraining_sla_breach import ModelTrainer
from evaluation.model_evaluation.evaluation import evaluate_and_save_best_model


def run_null_handling(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔹 Running Null Handling...")
    processor = DataProcessor(df)
    processor.remove_high_null_columns()
    processor.handle_missing_values()
    return processor.df_cleaned


def run_time_series_processing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔹 Running Time Series Feature Engineering...")
    processor = TimeSeriesProcessor(df, reference_date_col='created_date')
    return processor.process()


def run_feature_encoding(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔹 Running Feature Encoding...")
    encoder = FeatureEncoder(df, target_col='SLA Breach', verbose=True)
    df_encoded, _, _ = encoder.encode()
    return df_encoded


def run_leak_removal_and_smote(df: pd.DataFrame):
    print("\n🔹 Running Leaky Feature Removal + SMOTE...")
    remover = LeakyFeatureRemover(target_col='SLA Breach', verbose=True)
    X, y = remover.fit_transform(df)

    smoter = SMOTEHandler(verbose=True)
    X_resampled, y_resampled = smoter.apply(X, y)

    return X_resampled, y_resampled


def train_evaluate_and_save(model_type, X_train, X_test, y_train, y_test, current_best_score=None):
    print(f"\n🔸 Training and Evaluating {model_type.upper()} Model...")
    trainer = ModelTrainer(model_type=model_type, verbose=True)
    model = trainer.train_with_gridsearch(X_train, y_train)

    model_path = f"models/best_{model_type}_model.pkl"

    # Evaluate and conditionally save
    best_score, saved = evaluate_and_save_best_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_save_path=model_path,
        current_best_score=current_best_score,
        metric="roc_auc"
    )

    return model_type, trainer.best_params, best_score, saved


import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

def full_pipeline():
    print("\n🚀 Starting Full Pipeline...")
    
    # Step 1: Load raw data
    df = pd.read_csv("data/raw/itsm_sla_tickets_dataset_extended.csv")
    print(f"📂 Raw data loaded. Shape: {df.shape}")
    
    # Step 2: Preprocessing
    df_clean = run_null_handling(df)
    df_time_features = run_time_series_processing(df_clean)
    df_encoded = run_feature_encoding(df_time_features)
    
    # Save encoded data
    df_encoded.to_csv("data/processed/encoded_data.csv", index=False)
    df_encoded = pd.read_csv("data/processed/encoded_data.csv")
    
    # Step 3: Feature filtering + balancing
    X, y = run_leak_removal_and_smote(df_encoded)
    
    # Step 4: Save feature names for inference
    print("\n Saving feature names for inference...")
    feature_names = list(X.columns)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save feature names
    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_names, f)
    
    print(f"Saved {len(feature_names)} feature names to models/feature_columns.json")
    print(f" Sample features: {feature_names[:5]}")
    
    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f" Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 6: Train, evaluate and save models in parallel
    model_types = ['rf', 'xgb']
    current_best_score = None
    results = {}
    
    print("\n Training models in parallel...")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                train_evaluate_and_save,
                model_type,
                X_train,
                X_test,
                y_train,
                y_test,
                current_best_score
            )
            for model_type in model_types
        ]
        
        for future in as_completed(futures):
            model_type, best_params, score, saved = future.result()
            results[model_type] = {
                "best_params": best_params,
                "roc_auc": score,
                "saved": saved
            }
            
            # Update best score if necessary
            if current_best_score is None or score > current_best_score:
                current_best_score = score
                print(f" New best score: {score:.4f} ({model_type.upper()})")
    
    # Step 7: Save training summary
    print("\Model Training and Evaluation Complete. Summary:")
    best_model = None
    best_score = 0
    
    for model, info in results.items():
        print(f"   - {model.upper()} | ROC AUC: {info['roc_auc']:.4f} | Saved: {info['saved']}")
        if info['roc_auc'] > best_score:
            best_score = info['roc_auc']
            best_model = model
    
    # Step 8: Save training metadata
    training_metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "feature_count": len(feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_model": best_model,
        "best_score": best_score,
        "results": results,
        "feature_names_saved": True
    }
    
    with open("models/training_metadata.json", "w") as f:
        json.dump(training_metadata, f, indent=2)
    
    print(f"\n Best model: {best_model.upper()} with ROC AUC: {best_score:.4f}")
    print(" Training metadata saved to models/training_metadata.json")
    print("\nFull Pipeline Completed Successfully!")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'results': results,
        'best_model': best_model,
        'best_score': best_score
    }

if __name__ == "__main__":
    try:
        pipeline_results = full_pipeline()
        print(f"\n Pipeline completed with {len(pipeline_results['feature_names'])} features")
    except Exception as e:
        print(f" Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
