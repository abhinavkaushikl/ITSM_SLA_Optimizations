import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib  # for saving models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_and_save_best_model(model, X_test, y_test, model_save_path, 
                                 current_best_score=None, metric='roc_auc'):
    """
    Evaluate the model, print metrics, and save the model if it's the best so far.

    Parameters:
    - model: Trained sklearn-like model with predict and predict_proba methods.
    - X_test: Features for testing.
    - y_test: True labels for test.
    - model_save_path: Path to save the best model.
    - current_best_score: Previous best score to compare against. If None, saves current model.
    - metric: Metric used to determine best model. Supports 'roc_auc' or 'accuracy'.

    Returns:
    - best_score: The better score between current and previous best.
    - saved: Boolean, whether the model was saved.
    """
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except (AttributeError, IndexError):
        y_prob = None
        roc_auc = None

    acc = model.score(X_test, y_test)

    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    if roc_auc is not None:
        logger.info("ROC AUC Score: %.4f", roc_auc)
    else:
        logger.warning("ROC AUC not available - model does not support predict_proba or binary classification")

    # Decide which metric to use for saving
    if metric == 'roc_auc' and roc_auc is not None:
        score = roc_auc
    elif metric == 'accuracy':
        score = acc
    else:
        # fallback to accuracy if ROC AUC not available
        score = acc

    saved = False
    if current_best_score is None or score > current_best_score:
        joblib.dump(model, model_save_path)
        logger.info(f"Model saved to {model_save_path} with {metric} = {score:.4f}")
        best_score = score
        saved = True
    else:
        logger.info(f"Model not saved. Current best {metric}: {current_best_score:.4f} is better than {score:.4f}")
        best_score = current_best_score

    return best_score, saved
