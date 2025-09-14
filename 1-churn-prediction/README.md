# Assignment – Churn Prediction

### Due date: 9/23/2025

## Assignment Instructions & Grading Rubric

### 1) Purpose

Follow the class pipeline (Steps 1–6) to build a **reproducible churn prediction workflow**: preprocessing → EDA → feature engineering/modeling → prediction/evaluation.
**Deliverable:** a single, reproducible **Colab notebook (`.ipynb`)** submitted on Canvas.

---

### 2) Files & Data

* **Starter notebook:** `Churn_Prediction_Assignment#1.ipynb` (on Canvas)
* **Reference notebook:** `S2_Churn.ipynb` (for ideas)
* **Dataset:** `Telco-Customer-Churn.csv` (place in the same folder as your notebook)
* **Target variable:** `Churn` (Yes/No → 1/0)

---

### 3) Tasks & Points — Total 9 points

**Scoring:** Each item is graded as **met (1 point)** or **not met (0 points)**.
**Minimum deduction unit:** 1 point (no 0.5-point deductions).

#### Q1 — Steps 1 & 2: Preprocessing & Variable Setup (**1 point**)

* [ ] Load the CSV into **`df`**.
* [ ] Coerce **`TotalCharges`** to numeric (force conversion) and **impute missings with mean**.
* [ ] Convert **`Churn`** to **0/1** (Yes=1, No=0).
* [ ] **Manually create** four dummies (do **not** one-hot encode these):

  * `gender_Female`
  * `PhoneService_Yes`
  * `InternetService_Fiber optic`
  * `Contract_Two year`
* [ ] Show `df.info()` and `df.head()` (first 5 rows) for sanity check.

#### Q2 — Step 3: EDA (**2 points**)

* [ ] For **≥1 numeric variable**: show a **distribution plot** (e.g., histogram) **with basic statistics**.
* [ ] Report **overall churn rate** and **churn rate by ≥1 categorical variable** (e.g., `Contract` or `PaymentMethod`) using a **crosstab/bar chart**.
* [ ] Provide **2–3 bullet observations** summarizing insights from EDA.

#### Q3 — Step 4: Feature Engineering & Modeling (**3 points**)

* [ ] Build **X with ≥8 features** (mix of numeric + dummies) including the **4 manual dummies** from Q1.
* [ ] Fit **`StandardScaler` on the training set only**; apply the **same fitted scaler** to both train and test.
* [ ] Fit a **logistic model with `statsmodels`** (GLM with `family=Binomial`) **including an intercept**.
* [ ] List the selected features in a short **Markdown cell**.

#### Q4 — Steps 5 & 6: Prediction, Evaluation & Feature Importance (**3 points**)

* [ ] Predict **probabilities** for the test set; convert to **classes at threshold 0.5**.
* [ ] Print the **classification report** and **confusion matrix**; compute **AUC**.
* [ ] **Plot the ROC curve** (axis labels; legend with AUC recommended).
* [ ] Write a **3–5 sentence interpretation** about metric trade-offs (e.g., precision–recall, FP vs. FN costs).

---

### 4) Code & Library Rules

* **Recommended imports:**

  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

  import statsmodels.api as sm
  ```
* The **4 dummies in Q1 must be created manually** (no automatic one-hot encoding for them).
* Ensure **reproducibility**: `train_test_split(..., random_state=42)`.
* **All plots** must have **clear titles, axis labels,** and **legends** where appropriate.

---

### 5) Submission Format

* **File name:** `LastName_FirstName_A1_Churn.ipynb`
* Add a **Markdown header** at the top with **your name, student ID, course,** and **assignment title**.
* Submit on **Canvas → Assignments → A#1**.

---

### 6) Timeline & Late Submission

* **Due:** **9/23, 11:59 pm**.
* **Late policy:** Up to **3 days late** allowed, **–1 point per day**.

  * Example: **3 days late → max score = 6 points** (even if fully correct).

---

### 7) Academic Integrity & Collaboration

* **Plagiarism** or **academic dishonesty** is strictly prohibited.
* Discussion is encouraged, but **write your own code and analysis**.
* If you collaborated at the idea level, **list collaborators and scope** at the top of your notebook.

---

## Grading Rubric (Checklist — 9 points total, no partial points)

> Each line is either **Yes=1** (meets all criteria) or **No=0**.

| Criterion ID | Item (Yes=1 / No=0)                                                                                                                                     | Notes                       |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **Q1-1**     | Correct numeric coercion for `TotalCharges` + mean imputation; `Churn` mapped to 0/1; **4 manual dummies created**; `df.info()` and `head()` displayed. | **Manual dummies only.**    |
| **Q2-1**     | Numeric distribution + **basic stats** shown **with proper labels**.                                                                                    |                             |
| **Q2-2**     | **Overall churn rate** and **churn-by-category** shown in **table/plot** + **2–3 bullet insights**.                                                     |                             |
| **Q3-1**     | **≥8 features** (numeric + dummies); includes the **4 manual dummies**.                                                                                 |                             |
| **Q3-2**     | **Scaler fitted on train only**; same scaler applied to train and test.                                                                                 |                             |
| **Q3-3**     | **GLM (Binomial/Logit) fit with intercept** using `statsmodels.api as sm`.                                                                              |                             |
| **Q4-1**     | Test **probabilities → class at 0.5**; **classification report** + **confusion matrix** printed.                                                        | Check threshold use.        |
| **Q4-2**     | **AUC reported** and **ROC curve plotted**.                                                                                                             |                             |
| **Q4-3**     | **3–5 sentence trade-off interpretation**.                                                                                                              | Evaluate reasoning quality. |

---

## Student Submission Checklist

* [ ] File name follows convention: `LastName_FirstName_A1_Churn.ipynb`.
* [ ] Top-of-notebook header with **name / ID / course / assignment**.
* [ ] Includes **all four parts**: preprocessing, EDA, modeling, evaluation **+ interpretation**.
* [ ] Submitted on **Canvas → Assignments → A#1** **before the deadline**.


---

### Convert .py to .ipynb using
```bash
jupytext --to ipynb churn-prediction.py
```

### Convert .ipynb to .py using
```bash
jupytext --to py churn-prediction.ipynb
```