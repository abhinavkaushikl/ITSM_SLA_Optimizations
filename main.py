from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import traceback
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Load model
try:
    model = joblib.load("models/best_rf_model.pkl")
    print(" Model loaded successfully")
except Exception as e:
    print(f" Failed to load model: {e}")
    model = None

app = FastAPI(
    title="SLA Breach Prediction API",
    description="Conservative SLA Breach Prediction with 75% Threshold",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class Ticket(BaseModel):
    created_date: str = "2024-01-15 10:30:00"
    incident_type: str = "Incident"
    priority: str = "P1"
    category: str = "Software"
    status: str = "Open"
    escalation_level: str = "Level 1"
    service_name: str = "Internet"
    service_category: str = "Core Network"
    customer_segment: str = "Enterprise"
    contract_type: str = "Standard"
    region: str = "North"
    business_unit: str = "Corporate"
    technician_skill_level: str = "Senior"
    ticket_source: str = "Portal"
    impact_level: str = "High"
    reopened_count: int = 0
    change_request_linked: str = "No"
    problem_ticket_linked: str = "No"

class BatchTickets(BaseModel):
    tickets: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {
        "message": "Conservative SLA Breach Prediction API",
        "model_loaded": model is not None,
        "status": "ready",
        "version": "2.0.0",
        "threshold": "75% (Conservative)",
        "preprocessing": "Uses training pipeline preprocessing",
        "endpoints": {
            "health": "/",
            "test": "/test",
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "model_info": "/model/info",
            "conservative_test": "/test/conservative",
            "no_breach_test": "/test/no-breach-scenarios",
            "additional_no_breach_test": "/test/additional-no-breach",
            "api_docs": "/docs"
        }
    }

@app.get("/test")
def test_endpoint():
    """Basic test endpoint with conservative threshold"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        from Inference_pipeline import predict_with_full_analysis
        
        # Test data
        test_data = {
            "created_date": "2024-01-15 10:30:00",
            "incident_type": "Incident",
            "priority": "P1",
            "category": "Software",
            "status": "Open",
            "escalation_level": "Level 2",
            "service_name": "Internet",
            "service_category": "Core Network",
            "customer_segment": "Enterprise",
            "contract_type": "Premium",
            "region": "North",
            "business_unit": "Corporate",
            "technician_skill_level": "Senior",
            "ticket_source": "Portal",
            "impact_level": "High",
            "reopened_count": 1,
            "change_request_linked": "No",
            "problem_ticket_linked": "No"
        }
        
        print(" Starting conservative threshold test...")
        
        # Use the new full analysis function
        analysis = predict_with_full_analysis(model, test_data, threshold=0.75)
        
        result = {
            "test_data": test_data,
            "analysis": analysis,
            "message": "Conservative threshold test successful",
            "threshold_used": 0.75
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Test error: {error_details}")
        return {
            "error": str(e),
            "message": "Conservative test failed",
            "details": error_details.split('\n')[-3:-1]
        }

@app.get("/test/additional-no-breach")
def additional_no_breach_test():
    """Test conservative threshold with scenarios that should NOT breach SLA"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    no_breach_scenarios = [
        {
            "name": "P4 Consumer Request - Very Low Risk",
            "expected": "No Breach",
            "data": {
                "created_date": "2024-01-15 14:00:00",
                "incident_type": "Request",
                "priority": "P4",
                "category": "Other",
                "status": "Open",
                "escalation_level": "Level 1",
                "service_name": "Internet",
                "service_category": "Access",
                "customer_segment": "Consumer",
                "contract_type": "Standard",
                "region": "East",
                "business_unit": "Retail",
                "technician_skill_level": "Senior",
                "ticket_source": "Portal",
                "impact_level": "Low",
                "reopened_count": 0,
                "change_request_linked": "No",
                "problem_ticket_linked": "No"
            }
        },
        {
            "name": "P3 Standard SME Ticket",
            "expected": "No Breach",
            "data": {
                "created_date": "2024-01-15 11:00:00",
                "incident_type": "Incident",
                "priority": "P3",
                "category": "Software",
                "status": "In Progress",
                "escalation_level": "Level 1",
                "service_name": "Cloud Services",
                "service_category": "Application",
                "customer_segment": "SME",
                "contract_type": "Standard",
                "region": "North",
                "business_unit": "Corporate",
                "technician_skill_level": "Senior",
                "ticket_source": "Email",
                "impact_level": "Medium",
                "reopened_count": 0,
                "change_request_linked": "No",
                "problem_ticket_linked": "No"
            }
        },
        {
            "name": "Already Resolved P2 Ticket",
            "expected": "No Breach",
            "data": {
                "created_date": "2024-01-14 10:00:00",
                "incident_type": "Incident",
                "priority": "P2",
                "category": "Software",
                "status": "Resolved",
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
        }
    ]
    
    results = []
    
    try:
        from Inference_pipeline import predict_with_full_analysis
        
        for scenario in no_breach_scenarios:
            try:
                print(f"🧪 Testing scenario: {scenario['name']}")
                
                # Use conservative threshold analysis
                analysis = predict_with_full_analysis(
                    model, 
                    scenario['data'], 
                    threshold=0.75
                )
                
                # Extract key metrics
                conservative_prediction = analysis['predictions']['conservative_prediction']
                standard_prediction = analysis['predictions']['standard_prediction']
                probability = analysis['predictions']['probability']
                risk_level = analysis['risk_assessment']['risk_level']
                
                # Check if prediction matches expectation
                expected_no_breach = scenario['expected'] == "No Breach"
                actual_no_breach = conservative_prediction == 0
                correct_prediction = expected_no_breach == actual_no_breach
                
                result_item = {
                    "scenario": scenario['name'],
                    "expected": scenario['expected'],
                    "conservative_prediction": conservative_prediction,
                    "standard_prediction": standard_prediction,
                    "probability": probability,
                    "risk_level": risk_level,
                    "correct_prediction": correct_prediction,
                    "prediction_label": "SLA Breach" if conservative_prediction == 1 else "No SLA Breach",
                    "threshold_benefit": "Reduced false positive" if (standard_prediction == 1 and conservative_prediction == 0) else "Same result",
                    "recommendation": analysis['risk_assessment']['recommendation'],
                    "input_summary": {
                        "priority": scenario['data']['priority'],
                        "impact": scenario['data']['impact_level'],
                        "type": scenario['data']['incident_type'],
                        "status": scenario['data']['status'],
                        "customer_segment": scenario['data']['customer_segment']
                    },
                    "status": "success"
                }
                
                results.append(result_item)
                
            except Exception as e:
                print(f" Scenario failed: {scenario['name']} - {e}")
                results.append({
                    "scenario": scenario['name'],
                    "error": str(e),
                    "status": "failed"
                })
        
        # Analysis
        successful_tests = [r for r in results if r.get('status') == 'success']
        correct_predictions = [r for r in successful_tests if r.get('correct_prediction', False)]
        conservative_no_breach = [r for r in successful_tests if r.get('conservative_prediction') == 0]
        standard_no_breach = [r for r in successful_tests if r.get('standard_prediction') == 0]
        false_positive_reduction = [r for r in successful_tests if r.get('threshold_benefit') == 'Reduced false positive']
        
        final_result = {
            "test_info": {
                "threshold_used": 0.75,
                "test_type": "Conservative No-Breach Scenarios",
                "total_scenarios": len(no_breach_scenarios)
            },
            "results": results,
            "analysis": {
                "successful_tests": len(successful_tests),
                "correct_predictions": len(correct_predictions),
                "accuracy_percentage": round(len(correct_predictions) / max(len(successful_tests), 1) * 100, 1),
                "conservative_no_breach_count": len(conservative_no_breach),
                "standard_no_breach_count": len(standard_no_breach),
                "false_positive_reduction_count": len(false_positive_reduction),
                "average_probability": round(
                    sum(r.get('probability', 0) for r in successful_tests) / max(len(successful_tests), 1), 3
                )
            },
            "threshold_benefits": {
                "reduced_false_positives": len(false_positive_reduction),
                "conservative_accuracy": round(len(conservative_no_breach) / max(len(successful_tests), 1) * 100, 1),
                "standard_accuracy": round(len(standard_no_breach) / max(len(successful_tests), 1) * 100, 1),
                "improvement": f"{len(false_positive_reduction)} fewer false positives"
            },
            "summary": {
                "correctly_predicted_no_breach": [r['scenario'] for r in conservative_no_breach],
                "false_positive_reductions": [r['scenario'] for r in false_positive_reduction],
                "risk_levels": {level: len([r for r in successful_tests if r.get('risk_level') == level]) 
                              for level in ['VERY LOW', 'LOW', 'MEDIUM', 'MEDIUM-HIGH', 'HIGH', 'CRITICAL']},
                "model_performance": "Conservative threshold reduces false positives while maintaining accuracy"
            }
        }
        
        return convert_numpy_types(final_result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f" Additional no-breach test failed: {error_details}")
        raise HTTPException(status_code=500, detail=f"Conservative test failed: {str(e)}")

@app.post("/predict")
def predict(ticket: Ticket):
    """Predict SLA breach with conservative 75% threshold"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        from Inference_pipeline import predict_with_full_analysis
        
        # Use full analysis with conservative threshold
        analysis = predict_with_full_analysis(model, ticket.dict(), threshold=0.75)
        
        # Format response
        result = {
            "prediction": analysis['predictions']['conservative_prediction'],
            "prediction_standard": analysis['predictions']['standard_prediction'],
            "probability": analysis['predictions']['probability'],
            "threshold_used": 0.75,
            "prediction_label": "SLA Breach" if analysis['predictions']['conservative_prediction'] == 1 else "No SLA Breach",
            "risk_assessment": analysis['risk_assessment'],
            "recommendation": analysis['recommendation'],
            "input_summary": analysis['ticket_info'],
            "threshold_analysis": analysis.get('threshold_analysis', {}),
            "conservative_benefits": {
                "reduced_false_positives": analysis['predictions']['standard_prediction'] == 1 and analysis['predictions']['conservative_prediction'] == 0,
                "higher_confidence": "Only predicts breach when ≥75% confident"
            }
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f" Prediction error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Conservative prediction failed: {str(e)}")

@app.post("/predict/batch")
def predict_batch(batch_data: BatchTickets):
    """Batch prediction with conservative threshold"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    tickets = batch_data.tickets
    
    if not tickets:
        raise HTTPException(status_code=400, detail="No tickets provided")
    
    if len(tickets) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickets allowed per batch")
    
    try:
        from Inference_pipeline import predict_with_full_analysis
        
        results = []
        processing_stats = {
            "total_tickets": len(tickets),
            "successful": 0,
            "failed": 0,
            "conservative_breach": 0,
            "standard_breach": 0,
            "false_positive_reduction": 0,
            "risk_levels": {"VERY LOW": 0, "LOW": 0, "MEDIUM": 0, "MEDIUM-HIGH": 0, "HIGH": 0, "CRITICAL": 0}
        }
        
        for i, ticket_data in enumerate(tickets):
            try:
                # Full analysis for each ticket
                analysis = predict_with_full_analysis(model, ticket_data, threshold=0.75)
                
                conservative_pred = analysis['predictions']['conservative_prediction']
                standard_pred = analysis['predictions']['standard_prediction']
                risk_level = analysis['risk_assessment']['risk_level']
                
                # Update stats
                processing_stats["successful"] += 1
                if conservative_pred == 1:
                    processing_stats["conservative_breach"] += 1
                if standard_pred == 1:
                    processing_stats["standard_breach"] += 1
                if standard_pred == 1 and conservative_pred == 0:
                    processing_stats["false_positive_reduction"] += 1
                
                processing_stats["risk_levels"][risk_level] += 1
                
                result_item = {
                    "ticket_index": i,
                    "conservative_prediction": conservative_pred,
                    "standard_prediction": standard_pred,
                    "probability": analysis['predictions']['probability'],
                    "risk_level": risk_level,
                    "recommendation": analysis['recommendation']['action'],
                    "false_positive_avoided": standard_pred == 1 and conservative_pred == 0,
                    "input_summary": analysis['ticket_info'],
                    "status": "success"
                }
                
                results.append(result_item)
                
            except Exception as e:
                results.append({
                    "ticket_index": i,
                    "error": str(e),
                    "status": "failed"
                })
                processing_stats["failed"] += 1
        
        final_result = {
            "processing_stats": processing_stats,
            "results": results,
            "conservative_analysis": {
                "threshold_used": 0.75,
                "success_rate": round(processing_stats["successful"] / processing_stats["total_tickets"] * 100, 1),
                "conservative_breach_rate": round(processing_stats["conservative_breach"] / max(processing_stats["successful"], 1) * 100, 1),
                "standard_breach_rate": round(processing_stats["standard_breach"] / max(processing_stats["successful"], 1) * 100, 1),
                "false_positive_reduction": processing_stats["false_positive_reduction"],
                "alert_reduction_percentage": round(
                    (processing_stats["standard_breach"] - processing_stats["conservative_breach"]) / 
                    max(processing_stats["standard_breach"], 1) * 100, 1
                ),
                "average_probability": round(
                    sum(r.get('probability', 0) for r in results if r['status'] == 'success') / 
                    max(processing_stats["successful"], 1), 3
                )
            },
            "summary": {
                "total_alerts_reduced": processing_stats["standard_breach"] - processing_stats["conservative_breach"],
                "risk_distribution": processing_stats["risk_levels"],
                "model_performance": "Conservative threshold significantly reduces false positives"
            }
        }
        
        return convert_numpy_types(final_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
def model_info():
    """Get information about the conservative model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        model_type = type(model).__name__
        
        result = {
            "model_type": model_type,
            "model_loaded": True,
            "threshold_info": {
                "decision_threshold": 0.75,
                "threshold_type": "Conservative",
                "purpose": "Reduce false positives",
                "confidence_requirement": "75% minimum for breach prediction"
            },
            "model_attributes": {
                "has_predict": hasattr(model, 'predict'),
                "has_predict_proba": hasattr(model, 'predict_proba'),
                "has_feature_importances": hasattr(model, 'feature_importances_')
            },
            "api_version": "2.0.0 - Conservative Threshold"
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        return {
            "model_type": "Unknown",
            "model_loaded": True,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
