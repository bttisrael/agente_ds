# Model Metrics

**Type:** classification | **Target:** `hired`

## Model Comparison

|                           |   mean |    std |
|:--------------------------|-------:|-------:|
| GradientBoosting          | 0.7061 | 0      |
| LogisticRegression        | 0.7061 | 0      |
| LogisticRegression_Optuna | 0.7061 | 0      |
| Stacking_Optuna           | 0.7061 | 0      |
| GradientBoosting_Optuna   | 0.7061 | 0      |
| LightGBM                  | 0.706  | 0      |
| LightGBM_Optuna           | 0.706  | 0      |
| XGBoost                   | 0.702  | 0.0005 |
| RandomForest              | 0.7015 | 0.0007 |
| ExtraTrees                | 0.6953 | 0.0012 |

**Selected model:** `GradientBoosting`

**ACCURACY (test):** 0.7060

```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00     11758
           1       0.71      1.00      0.83     28242

    accuracy                           0.71     40000
   macro avg       0.35      0.50      0.41     40000
weighted avg       0.50      0.71      0.58     40000

```

## AI Interpretation

## Model Interpretation: Resume Screening Classification

### Model Selection Rationale

GradientBoosting emerged as the optimal choice, though the results reveal a critical warning sign: virtually all models achieved nearly identical performance (~0.706), with the top seven models showing zero standard deviation. This uniformity strongly suggests that **all models are simply predicting the majority class** (hired = 70.61%). The GradientBoosting accuracy of 0.7060 essentially matches the baseline class distribution, indicating the model has learned little to no discriminative patterns. This is further evidenced by the complete absence of variation across cross-validation folds (std=0.0000), which is statistically improbable for a genuinely learning model. The slight performance degradation in tree ensemble methods with higher variance (RandomForest: 0.7015, ExtraTrees: 0.6953) suggests these models attempted minimal class differentiation but struggled with the severe class imbalance that wasn't properly addressed during training.

### Business Context and Practical Meaning

In practical terms, a 70.6% accuracy means this model would correctly predict hiring outcomes for roughly 7 out of 10 candidates, but this metric is **dangerously misleading** for production use. If the model is predominantly predicting "hired" for all candidates, it would have near-zero precision for identifying candidates who should *not* be hired—potentially the more valuable business use case. For a resume screening system, this creates serious operational risk: the model would advance nearly all applicants to human review, providing no efficiency gains over manual screening. The organization would waste recruiter time reviewing unsuitable candidates while gaining false confidence in an "automated" system. More critically, without examining precision, recall, and F1-scores for both classes separately, we cannot determine if the model identifies any true negatives (correctly rejected candidates) at all.

### Critical Limitations and Points of Attention

Several data quality issues severely compromise model reliability. The negative resume word counts and mixed CGPA scales (exceeding 10.0) indicate fundamental data integrity problems that likely injected noise into model training. The extreme class imbalance (70:30 ratio) was clearly not mitigated through stratified sampling, SMOTE, or class weighting, causing model collapse to majority-class prediction. Additionally, the highly right-skewed experience distribution and sparse features (research papers, hackathons) suggest the model may have insufficient signal from rare but potentially predictive attributes. The lack of feature importance analysis leaves us blind to whether legitimate hiring signals exist in the data or if we're facing a fundamentally unpredictable target variable given available features.

### Production Deployment Recommendations

**Do not deploy this model to production in its current state.** Before any deployment consideration: (1) **Audit and clean the data**—investigate negative word counts, normalize CGPA by university tier, and validate all numeric ranges; (2) **Retrain with proper class imbalance handling**—implement stratified k-fold cross-validation, apply class weights inversely proportional to class frequencies, or use SMOTE for minority class oversampling; (3) **Evaluate with appropriate metrics**—report precision, recall, F1-score, and ROC-AUC for both classes, as accuracy is meaningless with imbalanced data; (4) **Conduct feature engineering**—create interaction terms between sparse features and experience, normalize skewed distributions, and generate domain-specific features (e.g., prestige scores combining CGPA and university tier). Only after achieving genuine minority class recall above 60% and confirming the model isn't simply predicting the majority class should deployment be considered, paired with human-in-the-loop validation for high-stakes hiring decisions.
