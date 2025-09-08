# Analytics Plan: Akron Zoo Membership Upgrade Prediction

Prepared with assistance from Claude 4.0 Sonnet LLM.

## Executive Summary
This plan outlines our approach to identify explainable and reliable predictors of customer membership upgrades at Akron Zoo. We will develop and compare multiple classification models to achieve four key objectives: predict upgrade probability, identify managerially useful features, quantify predictive effects, and extract actionable business insights.

## 1. Business Problem Definition

**Primary Goal:** Identify features of the membership program and customer characteristics that reliably predict membership upgrades at annual renewal.

**Success Criteria:**
- High predictive accuracy (>80% on test set)
- Model interpretability for management decisions
- Robustness across different data subsets
- Actionable insights for increasing upgrade rates

## 2. Data Understanding & Preparation

### 2.1 Data Overview
- **Training Set:** 740 customers with 19 features + target
- **Test Set:** 303 customers for final validation
- **Target Variable:** UPD (1=Upgrade, 0=No Upgrade) - perfectly balanced
- **No missing values** - data quality is excellent

### 2.2 Feature Categories
1. **Customer Perception Features (8):** benefits, costs, value, identity, know, sat, fle, trustfor
   - Already standardized (mean≈0, std≈1)
   - Represent subjective evaluations of membership experience

2. **Demographic Features (6):** age_rec, gender, educ/educnew, mstat, size, child1
   - Categorical/ordinal variables
   - May require encoding for certain models

3. **Behavioral Features (2):** dist, tvis
   - Distance from zoo and visit frequency
   - Directly observable customer behaviors

### 2.3 Data Preparation Tasks
- **Feature Engineering:** Create interaction terms between perception and demographic variables
- **Encoding:** Implement appropriate encoding for categorical variables per model requirements
- **Validation Strategy:** Use stratified k-fold cross-validation to maintain target balance
- **Feature Selection:** Test various selection techniques to identify optimal feature sets

## 3. Exploratory Data Analysis Plan

### 3.1 Univariate Analysis
- Distribution analysis for all features
- Target variable correlation analysis
- Outlier detection and treatment strategy

## 4. Model Development Strategy

### 4.1 Model Portfolio
We will implement and compare five models, each offering different bias-variance tradeoffs:

1. **Logistic Regression (with Regularization)**
   - **Strengths:** High interpretability, coefficient significance, handles multicollinearity
   - **Approach:** Test L1, L2, and Elastic Net regularization
   - **Business Value:** Direct coefficient interpretation for feature impact

2. **Support Vector Machine (SVM)**
   - **Strengths:** Robust to outliers, effective with high-dimensional data
   - **Approach:** Test linear and RBF kernels with grid search optimization
   - **Business Value:** Handles complex non-linear relationships

3. **Naive Bayes**
   - **Strengths:** Fast, handles small datasets well, probabilistic output
   - **Approach:** Gaussian NB for continuous features, appropriate variants for categorical
   - **Business Value:** Provides probability scores for upgrade likelihood

4. **Random Forest**
   - **Strengths:** Feature importance ranking, handles interactions, robust to overfitting
   - **Approach:** Optimize tree depth, number of estimators, and feature sampling
   - **Business Value:** Natural feature importance for management insights

5. **Gradient Boosting Machine (GBM)**
   - **Strengths:** High accuracy, handles complex patterns, feature importance
   - **Approach:** Test XGBoost and LightGBM with hyperparameter optimization
   - **Business Value:** Often achieves highest predictive performance

### 4.2 Hyperparameter Optimization
- **Strategy:** Randomized search followed by grid search refinement
- **Cross-validation:** 5-fold stratified CV for robust evaluation
- **Metrics:** Focus on AUC-ROC, precision, recall, and F1-score

## 5. Model Evaluation Framework

### 5.1 Performance Metrics
- **Primary:** AUC-ROC (accounts for class probability)
- **Secondary:** Precision, Recall, F1-Score, Accuracy
- **Business Metric:** Cost-benefit analysis of upgrade predictions

### 5.2 Robustness Testing
- **Cross-validation:** 5-fold stratified CV with statistical significance testing
- **Bootstrap Sampling:** 1000 bootstrap samples for confidence intervals
- **Subset Analysis:** Model performance across demographic segments
- **Sensitivity Analysis:** Feature perturbation impact on predictions

### 5.3 Model Selection Criteria
1. **Accuracy:** Test set performance with confidence intervals
2. **Robustness:** Consistent performance across CV folds and subgroups
3. **Interpretability:** Ability to extract actionable business insights
4. **Bias Assessment:** Fair performance across customer segments

## 6. Model Interpretation & Business Insights

### 6.1 Feature Importance Analysis
- **Global Importance:** Overall feature rankings across models
- **Local Explanations:** SHAP values for individual predictions
- **Interaction Effects:** Two-way and three-way feature interactions
- **Threshold Analysis:** Identify actionable cut-points for key features

### 6.2 Business Impact Quantification
- **Upgrade Probability Modeling:** Probability calibration for reliable estimates
- **Feature Effect Sizes:** Quantify impact of 1-unit changes in key variables
- **Scenario Analysis:** Model outcomes under different feature improvement strategies
- **ROI Calculations:** Expected value of targeting high-probability customers

### 6.3 Actionable Insights Generation
- **Customer Segmentation:** Identify distinct upgrade-prone customer profiles
- **Intervention Strategies:** Specific recommendations for increasing upgrade rates
- **Counter-intuitive Findings:** Investigate and explain unexpected patterns
- **Competitive Advantage:** Strategic insights unique to Akron Zoo's context

## 7. Validation & Testing Strategy

### 7.1 Internal Validation
- **Holdout Validation:** 20% of training data for model selection
- **Time-based Splits:** If temporal patterns exist in data
- **Stratified Sampling:** Maintain demographic and behavioral balance

### 7.2 Final Model Testing
- **Test Set Evaluation:** One-time evaluation on unseen test data
- **Confidence Intervals:** Statistical significance of performance differences
- **Error Analysis:** Detailed analysis of misclassified cases
- **Threshold Optimization:** Business-optimal decision thresholds

## 8. Implementation Plan

### 8.1 Phase 1: Data Preparation & EDA
- Complete comprehensive EDA
- Feature engineering and selection
- Data quality validation

### 8.2 Phase 2: Model Development
- Implement all five models
- Hyperparameter optimization
- Cross-validation evaluation

### 8.3 Phase 3: Model Selection & Interpretation
- Comparative analysis and model selection
- Business insight extraction
- Final model validation on test set

### 8.4 Phase 4: Documentation & Presentation
- Technical documentation with statistical reasoning
- Management presentation preparation
- Implementation recommendations

## 9. Expected Deliverables

### 9.1 Technical Outputs
- **Best Model:** Optimized classifier meeting all success criteria
- **Feature Rankings:** Quantified importance of all predictive features
- **Performance Metrics:** Comprehensive evaluation with confidence intervals
- **Prediction Probabilities:** Calibrated upgrade probabilities for test set

### 9.2 Business Outputs
- **Customer Profiles:** Detailed segments with upgrade propensities
- **Action Plan:** Specific recommendations for improving upgrade rates
- **ROI Projections:** Expected financial impact of implementing insights
- **Strategic Insights:** Competitive advantages and market positioning opportunities

## 10. Risk Mitigation

### 10.1 Technical Risks
- **Overfitting:** Addressed through rigorous cross-validation and regularization
- **Feature Selection Bias:** Multiple selection methods and stability testing
- **Model Generalization:** Conservative evaluation and robustness testing

### 10.2 Business Risks
- **Insight Actionability:** Continuous validation with domain knowledge
- **Implementation Feasibility:** Practical constraints consideration in recommendations
- **Ethical Considerations:** Fair treatment across customer demographics

## Conclusion
This comprehensive analytics plan ensures we deliver a robust, interpretable, and actionable solution to Akron Zoo's membership upgrade prediction challenge. By systematically comparing multiple models and focusing on business interpretability, we will identify the most reliable predictors while providing clear guidance for strategic decision-making.
