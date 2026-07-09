import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment


def train_model(df, target_col,exp):
        """Trains machine learning model"""
    
        exp.setup(df, target=target_col)


        best_model = exp.compare_models(
                verbose=False, 
                turbo=True,
                exclude=["lightgbm", "catboost", "xgboost"] # excluded to avoid OpenMP dependency on macOS
                )
        print(type(best_model))
        print(best_model)
        leaderboard = exp.pull()

        X = exp.get_config("X_train")
        y = exp.get_config("y_train")

        best_metrics = leaderboard.iloc[0].to_dict()

        return {
                "best_model": best_model,
                "leaderboard": leaderboard,
                "best_metrics": best_metrics,
                "X": X,
                "y": y
        }

    
