# Model Evaluation

## `GradientBoosting`
**Type:** classification | **Target:** `hired`

| Dataset   | Accuracy |
|-----------|-------|
| Train     | 0.7061 |
| Test      | 0.7060 |
| Gap       | 0.0000  |

## AI Diagnostic

This model shows **underfitting** characteristics. While the training and test accuracy are virtually identical (indicating good generalization with no overfitting), the absolute performance of ~70.6% is mediocre for both sets. This suggests the model hasn't captured enough complexity in the data patterns to make strong predictions about hiring decisions.

To improve performance, you should increase model complexity by tuning hyperparameters such as increasing `n_estimators`, `max_depth`, or `learning_rate`. Also consider feature engineering to create more informative predictors, checking for class imbalance issues, or trying other algorithms. The near-zero gap is good news—it means once you find better features or parameters, the improvements should generalize well to new data.

## Optimized Parameters (Optuna)
```json
{
  "n_estimators": 256,
  "learning_rate": 0.026367402653214092,
  "max_depth": 2,
  "subsample": 0.5721011661344177
}
```
