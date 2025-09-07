# Assignment – Akron Zoo

**Draft Due Date:** 09/10/2024  
**Final Due Date:** 09/17/2024  

---

## Business Problem
The Akron Zoo needs to **identify the features of the membership program and customer characteristics** (or combinations thereof) that are explainable and reliable predictors of a customer upgrading their membership at the time of the annual renewal.

The Akron Zoo Society has compiled a dataset of its customers including:
- Evaluation/use of membership  
- Demographic characteristics  
- Upgrade decision (yes/no)  

---

## Background
During the program, you have learned several predictive/classification models that vary in terms of bias-variance tradeoffs and statistical approaches. You are also knowledgeable about:

- (a) Validating and evaluating a predictive model  
- (b) Checking for sensitivity and robustness  
- (c) Using multiple model selection criteria  
- (d) Interpreting model parameter estimates to provide insights for solving the business problem  

Some of the models you are familiar with include:
- Logistic/Regularized Regression  
- Support Vector Machines (SVM)  
- Naïve Bayes  
- Random Forest  
- Gradient Boosting Machines (GBM)  

---

## Objectives
Your goal is to use these models to:  

1. **Predict** a customer’s probability of upgrading.  
2. **Identify** features (or combinations) that are managerially useful to increase this probability.  
3. **Quantify** the predictive effect of these identified features.  
4. **Explain** any insights (including counter-intuitive ones) that your results uncover.  

You must compare the different models on the above 4 dimensions and recommend the **“best” model** that meets the criteria of:
- Robustness  
- Minimum bias  
- High accuracy  

Finally, thoroughly **explain and interpret the recommended model** in solving the business problem in a way that builds competitive advantage.  

---

## Deliverables
You are given two datasets: **training and test data** with identical features as described below.  

Each assignment requires 3 parts to be completed and submitted:

### 1. Python Code-book (60% Grade)
- Fully annotated code that addresses and solves the analytics problem.  
- Annotation must explain the **statistical reasoning** for key steps.  

### 2. Presentation Slides (40% Grade)
- Maximum 12 widescreen slides.  
- Present your project as if delivering it to a senior manager.  
- The manager is statistically knowledgeable but focused on **accuracy, robustness, and actionable insights**.  

### 3. Analytics Plan
- A written plan describing your approach to solving the business problem.  
- This plan should have guided your work in **#1** above.  

---

## Data Description

**Target Variable**  
- `UPD`: 1 = Upgrade, 0 = No Upgrade  

**Customer Membership Perception Features**  
- `BENEFITS`: Membership benefits as per customer  
- `COSTS`: Perceived cost of membership  
- `VALUE`: Perceived value of being a member  
- `IDENTITY`: Degree to which customer identifies with the organization  
- `KNOW`: Customer’s knowledge about the organization  
- `SAT`: Customer’s satisfaction with the membership  
- `FLE`: Customer’s satisfaction with frontline employees  
- `TRUSTFOR`: Customer’s trust in the organization  

**Demographic Features**  
- `AGE_REC`:  
  - 1 = 18–34  
  - 2 = 35–44  
  - 3 = 45–54  
  - 4 = 54+  

- `GENDER`:  
  - 1 = Male  
  - 2 = Female  

- `EDUC_REC`:  
  - 1 = Some college  
  - 2 = College degree  
  - 3 = Graduate school  

- `MSTAT`:  
  - 1 = Married  
  - 2 = Single  
  - 3 = Divorced/Separated  
  - 4 = Widow/Widower  

- `SIZE`: Total number of people in the household  
- `CHILD1`: Total number of children/grandchildren  

**Behavioral Features**  
- `DIST`: Distance from the zoo  
  - 1 = < 10 minutes  
  - 2 = 10–20 minutes  
  - 3 = 21–30 minutes  
  - 4 = > 30 minutes  

- `TVIS`: Number of visits using membership  
  - 1 = ≤ 2  
  - 2 = 3–4  
  - 3 = 5–6  
  - 4 = 7–8  
  - 5 = > 8  

**Other**  
- `NEW ID`: Unique ID for each customer  

---
