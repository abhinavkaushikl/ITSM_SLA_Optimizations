from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, model_type='rf', random_state=42, verbose=True):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.best_params = {}

    def get_param_grid(self):
        if self.model_type == 'rf':
            return {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt'],
                'class_weight': ['balanced']
            }
        elif self.model_type == 'xgb':
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1]
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def get_model(self):
        if self.model_type == 'rf':
            return RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == 'xgb':
            return XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_with_gridsearch(self, X_train, y_train, cv=3, scoring='f1', n_jobs=-1):
        param_grid = self.get_param_grid()
        model = self.get_model()

        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=self.verbose)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        if self.verbose:
            print(f"Best Params for {self.model_type.upper()}: {self.best_params}")

        return self.model