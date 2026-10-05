"""Model zoo. LightGBM / XGBoost are optional imports so tests run without them."""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _factory(cls, **params):
    """Bind the class and params now (a bare lambda would look up `cls` late and
    pick up whichever model class was assigned last)."""
    return lambda: cls(**params)


def get_models(task: str, seed: int = 0, names=None) -> dict:
    """task: 'rul' (regression) or 'fail' (classification)."""
    reg = task == "rul"
    zoo = {
        "rf": lambda: (RandomForestRegressor if reg else RandomForestClassifier)(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )
    }
    try:
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if reg else lgb.LGBMClassifier
        zoo["lgbm"] = _factory(cls, n_estimators=400, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)
    except ImportError:
        pass
    try:
        import xgboost as xgb

        cls = xgb.XGBRegressor if reg else xgb.XGBClassifier
        zoo["xgb"] = _factory(cls, n_estimators=400, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    except ImportError:
        pass
    if names:
        zoo = {k: v for k, v in zoo.items() if k in names}
    return zoo
