import pandas as pd
from features.Missing_null_pipeline import DataProcessor
from features.handletimeseriesdata import TimeSeriesProcessor
from features.Encodingfeatures import FeatureEncoder
from features.leakageandsmote import LeakyFeatureRemover
import json
import os
import numpy as np
import joblib

def load_expected_features():
    """Load the feature names that were saved during training"""
    try:
        with open("models/feature_columns.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Feature columns file not found. Run training pipeline first.")
        return None

def load_model_safely(model_path="models/best_rf_model.pkl"):
    """
    Safely load model from pickle file, handling both dict and direct model formats
    """
    try:
        print(f"🔍 Loading model from: {model_path}")
        loaded_data = joblib.load(model_path)
        
        print(f"📦 Loaded data type: {type(loaded_data)}")
        
        # Check if it's a dictionary containing the model
        if isinstance(loaded_data, dict):
            print(f"📋 Dictionary keys: {list(loaded_data.keys())}")
            
            # Try different possible keys where the model might be stored
            possible_keys = ['model', 'best_model', 'trained_model', 'classifier', 'estimator']
            
            for key in possible_keys:
                if key in loaded_data:
                    model = loaded_data[key]
                    if hasattr(model, 'predict'):
                        print(f"✅ Found model in key '{key}': {type(model).__name__}")
                        return model
                    else:
                        print(f"⚠️ Object in key '{key}' is not a model: {type(model)}")
            
            # If no standard keys found, look for any object with predict method
            for key, value in loaded_data.items():
                if hasattr(value, 'predict'):
                    print(f"✅ Found model-like object in key '{key}': {type(value).__name__}")
                    return value
            
            print(f"❌ No model found in dictionary. Available keys: {list(loaded_data.keys())}")
            return None
        
        # Check if it's directly a model
        elif hasattr(loaded_data, 'predict'):
            print(f"✅ Loaded direct model: {type(loaded_data).__name__}")
            return loaded_data
        
        else:
            print(f"❌ Loaded object is not a model: {type(loaded_data)}")
            print(f"📋 Object attributes: {[attr for attr in dir(loaded_data) if not attr.startswith('_')][:10]}")
            return None
            
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"   ✅ {step_name} completed")
        return result
    except Exception as e:
        print(f"   ❌ {step_name} failed: {e}")
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
            print(f"   ⚠️ Feature '{col}' not in training features, skipping...")
    
    # Report alignment results
    missing_features = set(expected_features) - set(X.columns)
    print(f"   📊 Matched features: {matched_features}/{len(expected_features)}")
    print(f"   📝 Missing features filled with 0: {len(missing_features)}")
    print(f"   ✅ Final feature shape: {aligned_df.shape}")
    
    return aligned_df

def validate_model(model):
    """
    Validate that the model has required methods
    """
    if model is None:
        raise ValueError("❌ Model is None")
    
    if not hasattr(model, 'predict'):
        raise ValueError(f"❌ Model {type(model)} doesn't have predict method")
    
    print(f"✅ Model validation passed: {type(model).__name__}")
    return True

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
    print(f"🎯 Making predictions with {threshold*100}% confidence threshold...")
    
    # Validate model first
    validate_model(model)
    
    if hasattr(model, 'predict_proba'):
        try:
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
            
            print(f"   📊 Conservative predictions: {breach_predictions}/{total_predictions}")
            print(f"   📊 Standard predictions: {standard_predictions}/{total_predictions}")
            print(f"   📉 Alert reduction: {threshold_info['alert_reduction_percentage']}%")
            
            return predictions, probabilities, threshold_info
            
        except Exception as e:
            print(f"   ❌ predict_proba failed: {e}")
            # Fallback to standard predict
            predictions = model.predict(X)
            return predictions, None, {'threshold_used': 0.5, 'note': f'predict_proba failed: {e}'}
    else:
        # Fallback for models without predict_proba
        print("   ⚠️ Model doesn't support predict_proba, using standard predictions")
        predictions = model.predict(X)
        return predictions, None, {'threshold_used': 0.5, 'note': 'Standard threshold used - no predict_proba'}

def get_risk_assessment(probability, threshold=0.75):
    """
    Assess risk level based on probability and threshold
    
    Parameters:
    - probability: Predicted probability of SLA breach
    - threshold: Decision threshold used
    
    Returns:
    - risk_info: Dictionary with risk level and recommendations
    """
    # Handle case where probability is None (no predict_proba)
    if probability is None:
        probability = 0.5  # Default neutral probability
    
    if probability >= 0.95:
        risk_level = "CRITICAL"
        recommendation = "🚨 IMMEDIATE ESCALATION REQUIRED - Very high breach probability"
        priority = "P0"
    elif probability >= threshold:
        risk_level = "HIGH"
        recommendation = "⚠️ PRIORITY HANDLING - Exceeds confidence threshold"
        priority = "P1"
    elif probability >= 0.6:
        risk_level = "MEDIUM-HIGH"
        recommendation = "📋 MONITOR CLOSELY - Approaching threshold"
        priority = "P2"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
        recommendation = "👀 STANDARD MONITORING - Moderate risk"
        priority = "P3"
    elif probability >= 0.2:
        risk_level = "LOW"
        recommendation = "✅ STANDARD PROCESSING - Low risk"
        priority = "P4"
    else:
        risk_level = "VERY LOW"
        recommendation = "✅ ROUTINE PROCESSING - Very low risk"
        priority = "P4"
    
    return {
        'risk_level': risk_level,
        'recommendation': recommendation,
        'suggested_priority': priority,
        'probability': float(probability) if probability is not None else None,
        'threshold': threshold,
        'exceeds_threshold': probability >= threshold if probability is not None else False,
        'confidence_gap': float(probability - threshold) if probability is not None and probability >= threshold else float(threshold - probability) if probability is not None else 0
    }

def preprocess_for_prediction(ticket_data):
    """
    Complete preprocessing pipeline for inference
    """
    try:
        print(f"🚀 Starting preprocessing for ticket: {ticket_data.get('incident_type', 'Unknown')}")
        
        # Load expected features
        expected_features = load_expected_features()
        if expected_features is None:
            raise ValueError("Cannot load expected features. Run training pipeline first.")
        
        print(f"📋 Expected features: {len(expected_features)}")
        
        # Convert input to proper DataFrame format
        df = create_minimal_dataframe(ticket_data)
        print(f"📊 Created DataFrame with shape: {df.shape}")
        
        # Run preprocessing pipeline
        X = run_inference_preprocessing(df)
        print(f"📊 After preprocessing: {X.shape}")
        
        # Align with training features
        X_aligned = align_features_with_training(X, expected_features)
        
        print("✅ Preprocessing completed successfully!")
        return X_aligned
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
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
        
        # Validate model first
        validate_model(model)
        
        # Preprocess the ticket
        X = preprocess_for_prediction(ticket_data)
        
        # Make predictions with conservative threshold
        predictions, probabilities, threshold_info = predict_with_conservative_threshold(
            model, X, threshold
        )
        
        # Get standard prediction for comparison
        try:
            standard_prediction = model.predict(X)[0]
        except Exception as e:
            print(f"⚠️ Standard prediction failed: {e}")
            standard_prediction = predictions[0]
        
        # Risk assessment
        probability = probabilities[0] if probabilities is not None else None
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
                'monitoring_required': (probability or 0) >= 0.4,
                'escalation_required': (probability or 0) >= threshold
            }
        }
        
        print("✅ Full prediction analysis completed!")
        return analysis
        
    except Exception as e:
        print(f"❌ Full prediction analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

# Test the preprocessing with conservative threshold
if __name__ == "__main__":
    print("🧪 Testing preprocessing pipeline with conservative threshold...")
    
    # Test model loading first
    print("\n🔍 Testing model loading...")
    model = load_model_safely("models/best_rf_model.pkl")
    
    if model is None:
        print("❌ Could not load model, testing preprocessing only...")
        
        # Try alternative model paths
        alternative_paths = [
            "models/best_xgb_model.pkl",
            "models/rf_model.pkl",
            "models/xgb_model.pkl"
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                print(f"🔍 Trying alternative path: {path}")
                model = load_model_safely(path)
                if model is not None:
                    break
    
    if model is not None:
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        print(f"📋 Model methods: {[method for method in dir(model) if not method.startswith('_') and callable(getattr(model, method))][:10]}")
    
    # Test data for "Already Resolved P2 Ticket" scenario
    test_ticket_resolved = {
        "created_date": "2024-01-14 10:00:00",
        "incident_type": "Incident",
        "priority": "P2",
        "category": "Software",
        "status": "Resolved",  # This is the key - already resolved
        "escalation_level": "Level 1",
        "service_name": "Internet",
        "service_category": "Core Network",
        "customer_segment": "Enterprise",
        "contract_type": "Standard",
        "region": "East",
        "business_unit": "Corporate",
        "technician_skill_level": "Senior",
        "ticket_source": "Portal",
        "impact_level": "Medium",
        "reopened_count": 0,
        "change_request_linked": "No",
        "problem_ticket_linked": "No"
    }
    
    try:
        # Test preprocessing only
        print("\n📋 Testing 'Already Resolved P2 Ticket' preprocessing:")
        result = preprocess_for_prediction(test_ticket_resolved)
        print(f"✅ Preprocessing successful! Shape: {result.shape}")
        
        # Test with model if available
        if model is not None:
            print("\n🎯 Testing full analysis with resolved ticket...")
            
            try:
                analysis = predict_with_full_analysis(model, test_ticket_resolved, threshold=0.75)
                print(f"✅ Analysis successful!")
                print(f"   Conservative prediction: {analysis['predictions']['conservative_prediction']}")
                print(f"   Standard prediction: {analysis['predictions']['standard_prediction']}")
                print(f"   Risk level: {analysis['risk_assessment']['risk_level']}")
                print(f"   Probability: {analysis['predictions']['probability']}")
                print(f"   Status: {analysis['ticket_info']['status']}")
                
                # For resolved tickets, prediction should typically be 0 (no breach)
                if analysis['predictions']['conservative_prediction'] == 0:
                    print("✅ Correct prediction for resolved ticket!")
                else:
                    print("⚠️ Unexpected prediction for resolved ticket")
                    
            except Exception as e:
                print(f"❌ Full analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
