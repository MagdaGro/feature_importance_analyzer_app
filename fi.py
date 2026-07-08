import numpy as np
from sklearn.inspection import permutation_importance

def get_feature_importance(model, X, y):
        """
        Calculate feature importance for a given model.
        Parameters
        """

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            method = "feature_importances_"

        # Linear models
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)

            if importance.ndim > 1:
                importance = importance.mean(axis=0)

            method = "coef_"

        # Fallback: permutation importance
        else:
            result = permutation_importance(
                model,
                X,
                y,
                n_repeats=10,
                random_state=42
                )

            importance = result.importances_mean
            method = "permutation_importance"

        return {
            "method": method,
            "features": X.columns.tolist(),
            "importance": importance.tolist()
            }