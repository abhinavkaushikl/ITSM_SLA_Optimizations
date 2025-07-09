import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_and_save_best_model(model, X_test, y_test, model_save_path,
                                 current_best_score=None, metric='roc_auc',
                                 decision_threshold=0.75):
    """
    Evaluate the model with custom decision threshold, print metrics, and save the model if it's the best so far.
    
    Parameters:
    - model: Trained sklearn-like model with predict and predict_proba methods.
    - X_test: Features for testing.
    - y_test: True labels for test.
    - model_save_path: Path to save the best model.
    - current_best_score: Previous best score to compare against. If None, saves current model.
    - metric: Metric used to determine best model. Supports 'roc_auc', 'accuracy', 'precision', 'recall', 'f1'.
    - decision_threshold: Threshold for classification (default 0.75 for more conservative predictions).
    
    Returns:
    - best_score: The better score between current and previous best.
    - saved: Boolean, whether the model was saved.
    """
    
    # Get probabilities and apply custom threshold
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_threshold = (y_prob >= decision_threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_prob)
    except (AttributeError, IndexError):
        y_prob = None
        y_pred_threshold = model.predict(X_test)
        roc_auc = None
    
    # Original predictions (0.5 threshold)
    y_pred_original = model.predict(X_test)
    
    # Calculate metrics for both thresholds
    acc_original = model.score(X_test, y_test)
    acc_threshold = np.mean(y_pred_threshold == y_test)
    
    # Log results for both thresholds
    logger.info("="*60)
    logger.info(f"EVALUATION RESULTS (Threshold: {decision_threshold})")
    logger.info("="*60)
    
    logger.info("Original Classification Report (0.5 threshold):")
    logger.info("\n%s", classification_report(y_test, y_pred_original))
    
    logger.info(f"Custom Threshold Classification Report ({decision_threshold} threshold):")
    logger.info("\n%s", classification_report(y_test, y_pred_threshold))
    
    logger.info("Original Confusion Matrix (0.5 threshold):")
    logger.info("\n%s", confusion_matrix(y_test, y_pred_original))
    
    logger.info(f"Custom Threshold Confusion Matrix ({decision_threshold} threshold):")
    logger.info("\n%s", confusion_matrix(y_test, y_pred_threshold))
    
    if roc_auc is not None:
        logger.info("ROC AUC Score: %.4f", roc_auc)
    else:
        logger.warning("ROC AUC not available - model does not support predict_proba or binary classification")
    
    logger.info("Accuracy (0.5 threshold): %.4f", acc_original)
    logger.info("Accuracy (%.2f threshold): %.4f", decision_threshold, acc_threshold)
    
    # Calculate additional metrics for threshold predictions
    if y_prob is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision_thresh = precision_score(y_test, y_pred_threshold)
        recall_thresh = recall_score(y_test, y_pred_threshold)
        f1_thresh = f1_score(y_test, y_pred_threshold)
        
        logger.info("Precision (%.2f threshold): %.4f", decision_threshold, precision_thresh)
        logger.info("Recall (%.2f threshold): %.4f", decision_threshold, recall_thresh)
        logger.info("F1 Score (%.2f threshold): %.4f", decision_threshold, f1_thresh)
        
        # Show prediction distribution
        breach_predictions = np.sum(y_pred_threshold)
        total_predictions = len(y_pred_threshold)
        logger.info("Breach Predictions: %d/%d (%.1f%%)", 
                   breach_predictions, total_predictions, 
                   (breach_predictions/total_predictions)*100)
    
    # Decide which metric to use for saving (using threshold-based predictions)
    if metric == 'roc_auc' and roc_auc is not None:
        score = roc_auc
    elif metric == 'accuracy':
        score = acc_threshold
    elif metric == 'precision' and y_prob is not None:
        score = precision_thresh
    elif metric == 'recall' and y_prob is not None:
        score = recall_thresh
    elif metric == 'f1' and y_prob is not None:
        score = f1_thresh
    else:
        # fallback to threshold-based accuracy
        score = acc_threshold
    
    saved = False
    if current_best_score is None or score > current_best_score:
        # Save model with threshold information
        model_info = {
            'model': model,
            'decision_threshold': decision_threshold,
            'best_score': score,
            'metric_used': metric
        }
        joblib.dump(model_info, model_save_path)
        logger.info(f"Model saved to {model_save_path} with {metric} = {score:.4f} (threshold: {decision_threshold})")
        best_score = score
        saved = True
    else:
        logger.info(f"Model not saved. Current best {metric}: {current_best_score:.4f} is better than {score:.4f}")
        best_score = current_best_score
    
    return best_score, saved

def load_model_with_threshold(model_path):
    """
    Load model that was saved with threshold information
    """
    try:
        model_info = joblib.load(model_path)
        if isinstance(model_info, dict) and 'model' in model_info:
            return model_info['model'], model_info.get('decision_threshold', 0.5)
        else:
            # Old format - just the model
            return model_info, 0.5
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def predict_with_threshold(model, X, threshold=0.75):
    """
    Make predictions with custom threshold
    """
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
        return (y_prob >= threshold).astype(int), y_prob
    else:
        return model.predict(X), None
