import pandas as pd
from features.Missing_null_pipeline import DataProcessor
from features.handletimeseriesdata import TimeSeriesProcessor
from features.Encodingfeatures import FeatureEncoder
from features.leakageandsmote import LeakyFeatureRemover
import json
import os
import numpy as np

def load_expected_features():
    """Load the feature names that were saved during training"""
    try:
        with open("models/feature_columns.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Feature columns file not found. Run training pipeline first.")
        return None

def create_minimal_dataframe(ticket_data):
    """
    Create a minimal DataFrame with all required columns for the pipeline
    """
    # Base DataFrame with required columns
    df_dict = {
        # Time columns
        'created_date': ticket_data.get('created_date', '2024-01-15 10:30:00'),
        'Created At': ticket_data.get('created_date', '2024-01-15 10:30:00'),
        'Responded At': ticket_data.get('responded_at', '2024-01-15 11:30:00'),
        
        # ID columns
        'Ticket ID': f"PRED_{hash(str(ticket_data)) % 10000}",
        'Customer ID': ticket_data.get('customer_id', 'C001'),
        'Assigned Technician ID': ticket_data.get('assigned_technician_id', 'TECH001'),
        'Team ID': ticket_data.get('team_id', 'TEAM001'),
        
        # Categorical columns
        'Ticket Type': ticket_data.get('incident_type', 'Incident'),
        'Priority': ticket_data.get('priority', 'P1'),
        'Root Cause Category': ticket_data.get('category', 'Software'),
        'Status': ticket_data.get('status', 'Open'),
        'Escalation Level': ticket_data.get('escalation_level', 'Level 1'),
        'Service Name': ticket_data.get('service_name', 'Internet'),
        'Service Category': ticket_data.get('service_category', 'Core Network'),
        'Customer Segment': ticket_data.get('customer_segment', 'Enterprise'),
        'Contract Type': ticket_data.get('contract_type', 'Standard'),
        'Region': ticket_data.get('region', 'North'),
        'Business Unit': ticket_data.get('business_unit', 'Corporate'),
        'Technician Skill Level': ticket_data.get('technician_skill_level', 'Senior'),
        'Escalation Team ID': ticket_data.get('escalation_team_id', 'ESC1'),
        'Ticket Source': ticket_data.get('ticket_source', 'Portal'),
        'Impact Level': ticket_data.get('impact_level', 'High'),
        
        # Numeric columns
        'Reopened Count': ticket_data.get('reopened_count', 0),
        
        # Boolean columns (as strings)
        'Change Request Linked': ticket_data.get('change_request_linked', 'No'),
        'Problem Ticket Linked': ticket_data.get('problem_ticket_linked', 'No'),
        
        # Target column (will be removed later)
        'SLA Breach': 0
    }
    
    # Create DataFrame
    df = pd.DataFrame([df_dict])
    
    # Ensure proper data types
    df['Reopened Count'] = df['Reopened Count'].astype(int)
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['Created At'] = pd.to_datetime(df['Created At'])
    df['Responded At'] = pd.to_datetime(df['Responded At'])
    
    return df

def safe_preprocessing_step(step_name, func, *args, **kwargs):
    """Safely execute a preprocessing step with error handling"""
    try:
        print(f"   {step_name}...")
        result = func(*args, **kwargs)
        print(f"    {step_name} completed")
        return result
    except Exception as e:
        print(f"    {step_name} failed: {e}")
        raise

def run_inference_preprocessing(df):
    """
    Run the same preprocessing steps as training pipeline with error handling
    """
    print("🔄 Running inference preprocessing...")
    
    # Step 1: Minimal null handling for single prediction
    def minimal_null_handling(df):
        # Just fill any remaining nulls, don't remove columns
        df_filled = df.fillna({
            'Reopened Count': 0,
            'Priority': 'P3',
            'Status': 'Open',
            'Ticket Type': 'Incident'
        })
        # Fill remaining nulls with appropriate defaults
        for col in df_filled.columns:
            if df_filled[col].dtype == 'object':
                df_filled[col] = df_filled[col].fillna('Unknown')
            else:
                df_filled[col] = df_filled[col].fillna(0)
        return df_filled
    
    df_cleaned = safe_preprocessing_step(
        "Step 1: Null handling", 
        minimal_null_handling, 
        df
    )
    
    # Step 2: Time series processing
    df_with_time = safe_preprocessing_step(
        "Step 2: Time series processing",
        lambda x: TimeSeriesProcessor(x, reference_date_col='created_date').process(),
        df_cleaned
    )
    
    # Step 3: Feature encoding
    def safe_encoding(df):
        encoder = FeatureEncoder(df, target_col='SLA Breach', verbose=False)
        df_encoded, _, _ = encoder.encode()
        return df_encoded
    
    df_encoded = safe_preprocessing_step(
        "Step 3: Feature encoding",
        safe_encoding,
        df_with_time
    )
    
    # Step 4: Remove leaky features
    def safe_leaky_removal(df):
        remover = LeakyFeatureRemover(target_col='SLA Breach', verbose=False)
        X, y = remover.fit_transform(df)
        return X
    
    X = safe_preprocessing_step(
        "Step 4: Leaky feature removal",
        safe_leaky_removal,
        df_encoded
    )
    
    return X

def align_features_with_training(X, expected_features):
    """
    Ensure the features match exactly what the model expects
    """
    print("🔧 Aligning features with training data...")
    
    # Create a DataFrame with all expected features initialized to 0
    aligned_df = pd.DataFrame(0, index=X.index, columns=expected_features)
    
    # Fill in the features that exist in our processed data
    matched_features = 0
    for col in X.columns:
        if col in expected_features:
            aligned_df[col] = X[col]
            matched_features += 1
        else:
            print(f"  Feature '{col}' not in training features, skipping...")
    
    # Report alignment results
    missing_features = set(expected_features) - set(X.columns)
    print(f"   Matched features: {matched_features}/{len(expected_features)}")
    print(f"    Missing features filled with 0: {len(missing_features)}")
    print(f"   Final feature shape: {aligned_df.shape}")
    
    return aligned_df

def predict_with_conservative_threshold(model, X, threshold=0.75):
    """
    Make predictions with conservative 75% threshold to reduce false positives
    
    Parameters:
    - model: Trained model with predict_proba method
    - X: Features for prediction
    - threshold: Decision threshold (default 0.75 for conservative predictions)
    
    Returns:
    - predictions: Binary predictions using the threshold
    - probabilities: Raw probabilities from the model
    - threshold_info: Information about the threshold used
    """
    print(f" Making predictions with {threshold*100}% confidence threshold...")
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Calculate prediction statistics
        total_predictions = len(predictions)
        breach_predictions = np.sum(predictions)
        standard_predictions = np.sum(model.predict(X))
        
        threshold_info = {
            'threshold_used': threshold,
            'total_predictions': total_predictions,
            'breach_predictions_conservative': int(breach_predictions),
            'breach_predictions_standard': int(standard_predictions),
            'reduction_in_alerts': int(standard_predictions - breach_predictions),
            'alert_reduction_percentage': round(((standard_predictions - breach_predictions) / max(standard_predictions, 1)) * 100, 1),
            'average_probability': float(np.mean(probabilities)),
            'max_probability': float(np.max(probabilities)),
            'min_probability': float(np.min(probabilities))
        }
        
        print(f"    Conservative predictions: {breach_predictions}/{total_predictions}")
        print(f"    Standard predictions: {standard_predictions}/{total_predictions}")
        print(f"    Alert reduction: {threshold_info['alert_reduction_percentage']}%")
        
        return predictions, probabilities, threshold_info
    else:
        # Fallback for models without predict_proba
        print("    Model doesn't support predict_proba, using standard predictions")
        predictions = model.predict(X)
        return predictions, None, {'threshold_used': 0.5, 'note': 'Standard threshold used'}

def get_risk_assessment(probability, threshold=0.75):
    """
    Assess risk level based on probability and threshold
    
    Parameters:
    - probability: Predicted probability of SLA breach
    - threshold: Decision threshold used
    
    Returns:
    - risk_info: Dictionary with risk level and recommendations
    """
    if probability >= 0.95:
        risk_level = "CRITICAL"
        recommendation = " IMMEDIATE ESCALATION REQUIRED - Very high breach probability"
        priority = "P0"
    elif probability >= threshold:
        risk_level = "HIGH"
        recommendation = " PRIORITY HANDLING - Exceeds confidence threshold"
        priority = "P1"
    elif probability >= 0.6:
        risk_level = "MEDIUM-HIGH"
        recommendation = " MONITOR CLOSELY - Approaching threshold"
        priority = "P2"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
        recommendation = " STANDARD MONITORING - Moderate risk"
        priority = "P3"
    elif probability >= 0.2:
        risk_level = "LOW"
        recommendation = " STANDARD PROCESSING - Low risk"
        priority = "P4"
    else:
        risk_level = "VERY LOW"
        recommendation = " ROUTINE PROCESSING - Very low risk"
        priority = "P4"
    
    return {
        'risk_level': risk_level,
        'recommendation': recommendation,
        'suggested_priority': priority,
        'probability': float(probability),
        'threshold': threshold,
        'exceeds_threshold': probability >= threshold,
        'confidence_gap': float(probability - threshold) if probability >= threshold else float(threshold - probability)
    }

def preprocess_for_prediction(ticket_data):
    """
    Complete preprocessing pipeline for inference
    """
    try:
        print(f" Starting preprocessing for ticket: {ticket_data.get('incident_type', 'Unknown')}")
        
        # Load expected features
        expected_features = load_expected_features()
        if expected_features is None:
            raise ValueError("Cannot load expected features. Run training pipeline first.")
        
        print(f" Expected features: {len(expected_features)}")
        
        # Convert input to proper DataFrame format
        df = create_minimal_dataframe(ticket_data)
        print(f" Created DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Run preprocessing pipeline
        X = run_inference_preprocessing(df)
        print(f" After preprocessing: {X.shape}")
        
        # Align with training features
        X_aligned = align_features_with_training(X, expected_features)
        
        print("Preprocessing completed successfully!")
        return X_aligned
        
    except Exception as e:
        print(f" Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def predict_with_full_analysis(model, ticket_data, threshold=0.75):
    """
    Complete prediction pipeline with conservative threshold and full analysis
    
    Parameters:
    - model: Trained model
    - ticket_data: Dictionary with ticket information
    - threshold: Decision threshold (default 0.75)
    
    Returns:
    - Complete prediction analysis with risk assessment
    """
    try:
        print("🔍 Starting full prediction analysis...")
        
        # Preprocess the ticket
        X = preprocess_for_prediction(ticket_data)
        
        # Make predictions with conservative threshold
        predictions, probabilities, threshold_info = predict_with_conservative_threshold(
            model, X, threshold
        )
        
        # Get standard prediction for comparison
        standard_prediction = model.predict(X)[0] if hasattr(model, 'predict') else predictions[0]
        
        # Risk assessment
        probability = probabilities[0] if probabilities is not None else 0.5
        risk_info = get_risk_assessment(probability, threshold)
        
        # Compile full analysis
        analysis = {
            'ticket_info': {
                'priority': ticket_data.get('priority', 'Unknown'),
                'impact_level': ticket_data.get('impact_level', 'Unknown'),
                'incident_type': ticket_data.get('incident_type', 'Unknown'),
                'category': ticket_data.get('category', 'Unknown'),
                'customer_segment': ticket_data.get('customer_segment', 'Unknown'),
                'status': ticket_data.get('status', 'Unknown')
            },
            'predictions': {
                'conservative_prediction': int(predictions[0]),
                'standard_prediction': int(standard_prediction),
                'probability': probability,
                'threshold_used': threshold
            },
            'risk_assessment': risk_info,
            'threshold_analysis': threshold_info,
            'recommendation': {
                'action': risk_info['recommendation'],
                'priority': risk_info['suggested_priority'],
                'monitoring_required': probability >= 0.4,
                'escalation_required': probability >= threshold
            }
        }
        
        print("✅ Full prediction analysis completed!")
        return analysis
        
    except Exception as e:
        print(f"❌ Full prediction analysis failed: {e}")
        raise

def load_model_with_threshold(model_path):
    """
    Load model that was saved with threshold information
    """
    import joblib
    try:
        model_info = joblib.load(model_path)
        if isinstance(model_info, dict) and 'model' in model_info:
            return model_info['model'], model_info.get('decision_threshold', 0.75)
        else:
            # Old format - just the model
            return model_info, 0.75  # Default to conservative threshold
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Test the preprocessing with conservative threshold
if __name__ == "__main__":
    print("🧪 Testing preprocessing pipeline with conservative threshold...")
    
    # Test data - should be low risk
    test_ticket_low_risk = {
        "created_date": "2024-01-15 14:00:00",
        "incident_type": "Request",
        "priority": "P4",
        "category": "Software",
        "status": "Open",
        "impact_level": "Low",
        "customer_segment": "Consumer",
        "contract_type": "Standard",
        "escalation_level": "Level 1"
    }
    
    # Test data - should be high risk
    test_ticket_high_risk = {
        "created_date": "2024-01-15 23:30:00",  # After hours
        "incident_type": "Incident",
        "priority": "P1",
        "category": "Hardware",
        "status": "Open",
        "impact_level": "High",
        "customer_segment": "Enterprise",
        "contract_type": "Premium",
        "escalation_level": "Level 3",
        "reopened_count": 2
    }
    
    try:
        # Test preprocessing only
        print("\n📋 Testing LOW RISK ticket preprocessing:")
        result_low = preprocess_for_prediction(test_ticket_low_risk)
        print(f"✅ Low risk test successful! Shape: {result_low.shape}")
        
        print("\n📋 Testing HIGH RISK ticket preprocessing:")
        result_high = preprocess_for_prediction(test_ticket_high_risk)
        print(f"✅ High risk test successful! Shape: {result_high.shape}")
        
        print(f"\n📊 Sample features: {list(result_low.columns)[:10]}")
        print(f"📈 Sample values (low risk): {result_low.iloc[0, :5].tolist()}")
        print(f"📈 Sample values (high risk): {result_high.iloc[0, :5].tolist()}")
        
        # Test threshold functionality (mock)
        print("\n🎯 Testing threshold functionality...")
        
        class MockModel:
            def predict_proba(self, X):
                # Mock probabilities - low risk ticket gets low probability
                if len(X) == 1:
                    return np.array([[0.8, 0.2]])  # Low probability of breach
                return np.array([[0.3, 0.7]])  # High probability of breach
            
            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] > 0.5).astype(int)
        
        mock_model = MockModel()
        
        # Test conservative threshold
        predictions, probabilities, threshold_info = predict_with_conservative_threshold(
            mock_model, result_low, threshold=0.75
        )
        
        print(f"📊 Mock prediction results:")
        print(f"   Probability: {probabilities[0]:.3f}")
        print(f"   Conservative prediction: {predictions[0]}")
        print(f"   Threshold info: {threshold_info}")
        
        # Test risk assessment
        risk_info = get_risk_assessment(probabilities[0], threshold=0.75)
        print(f"📊 Risk assessment: {risk_info['risk_level']} - {risk_info['recommendation']}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
