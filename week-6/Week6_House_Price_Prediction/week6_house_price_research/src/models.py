
"""Model registry, training and hyperparameter optimization."""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

RANDOM_SEED = 42

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=180, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=RANDOM_SEED, n_estimators=160
        ),
    }

def train_models(models, X_train, y_train):
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted

def tune_gradient_boosting(X_train, y_train, n_iter=8, cv=3):
    """Tune the main tree model without requiring external services."""
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_SEED),
        param_distributions={
            "n_estimators": randint(80, 260),
            "learning_rate": uniform(0.02, 0.18),
            "max_depth": randint(2, 6),
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 8),
            "subsample": uniform(0.65, 0.35),
        },
        n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search
