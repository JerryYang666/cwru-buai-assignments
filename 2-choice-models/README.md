# A#2: Discrete Choice (Travel Mode MNL) — **9 points**

## 1) Purpose

Model individual route/mode choice using a **Multinomial Logit (MNL)** model. Work with the classic **TravelMode** dataset and deliver a **reproducible Colab notebook (`.ipynb`)** that clearly shows: **preprocessing → EDA → feature specification → model estimation → prediction & evaluation**.
**Single deliverable:** the notebook submitted on Canvas.

---

## 2) Files & Data

* **Starter notebook:** `Choice_Models_Assignment_2.ipynb` (Canvas)
* **Reference notebook:** `S3_Choice_Models.ipynb`
* **Dataset:** `TravelMode.csv`
* **Target:** `choice` (1 if the alternative was chosen for that individual, 0 otherwise)
* **Key alternative-varying variables:** `wait`, `travel`, `vcost`, `gcost`
* **Key individual-specific variables:** `income`, `size`

---

## 3) Tasks & Points — **Total 9 points**

Minimum deduction unit is **1 point** (no 0.5-point deductions). Each item is graded **met (full credit) or not met (0)**.

### Q1 — Preprocessing & Variable Setup (**1 point**)

* Load the dataset into a **long-format DataFrame** (one row per individual × mode).
* Ensure or create these exact columns:

  * `ids` (from individual), `alts` (from mode), `choice` (1/0).
* **Manually create dummies & interactions** (**> 8 interaction terms**).
* Show `df.info()` and the **first 5 rows**.

### Q2 — EDA (**2 points**)

* Show **descriptive statistics** and at least **one distribution plot** (clearly labeled).
* Report **overall mode shares** (air/train/bus/car) **and** mode share **by grouping**. Present as **a table** and a **labeled bar chart**.
* Write **2–3 bullet observations**.

### Q3 — Feature Specification & MNL Estimation (**3 points**)

* Estimate an **MNL** including **Alternative-Specific Constants (ASCs)**.
* Include the four alternative-varying variables **`wait`, `travel`, `vcost`, `gcost`** with **sensible signs** (expect **negative** for disutility variables).
* Add at least one **individual-specific interaction** (e.g., `income × air`).

### Q4 — Interpretation (**3 points**)

* **Explain the meaning** of **each explanatory variable** reported in the results table and **derive insights** based on these interpretations.
* Provide a **3–5 sentence interpretation** of the **coefficients**.
* Discuss **model fit** (e.g., report **R²**) and **convergence**.

---

## 4) Code & Library Rules

* **All plots** must have **clear titles, axis labels, and legends** (where appropriate).
* Notebook must **run top-to-bottom without errors** (clean execution).

---

## 5) Submission Format

* **Filename:** `LastName_FirstName_A2_ChoiceModels.ipynb`
* **Top markdown header** must include: **your name, student ID, course, assignment title**.
* **Submit on Canvas:** *Assignments → A#2*.

---

## 6) Timeline & Late Submission

* **Due date:** See the **Canvas A#2** item for the exact deadline.
* **Late policy:** Up to **3 days late** allowed. **−1 point per day late**.

  * *Example:* If 3 days late, the **maximum score is 6 points**, even if everything is correct.

---

## 7) Academic Integrity & Collaboration

* **Plagiarism or academic dishonesty is strictly prohibited.**
* Discussion is encouraged, but you must **write your own code and analysis**.
* If you **collaborated (idea-level discussion)**, list **collaborators and scope** at the **top of your notebook**.

---

## Grading Rubric (Checklist — **9 points total; no partial points**)

| Item     | Criterion (Yes = 1 / No = 0)                                                                                                                                     | Notes   |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **Q1-1** | Data loaded; `ids`/`alts`/`choice` defined; **exactly one `choice=1` per `ids`**; **4 rows per `ids`**                                                           | 1 point |
| **Q2-1** | **At least 2 plots** shown with **proper labels**                                                                                                                | 1 point |
| **Q2-2** | **2–3 bullet insights** provided                                                                                                                                 | 1 point |
| **Q3-1** | **MNL fitted with ASCs**; **base alternative documented**; `wait`, `travel`, `vcost`, `gcost` included                                                           | 1 point |
| **Q3-2** | **At least 8 total features** including **individual-specific interaction** (e.g., `income×air`) **included and explained**; features **documented in markdown** | 1 point |
| **Q3-3** | **Convergence achieved**; **R² reported**                                                                                                                        | 1 point |
| **Q4-1** | **Model fit interpreted**                                                                                                                                        | 1 point |
| **Q4-2** | Explain the **meaning of each explanatory variable** from the results table; **derive insights**                                                                 | 1 point |
| **Q4-3** | Provide **3–5 sentence interpretation** of **coefficients**                                                                                                      | 1 point |

**Scoring principle:** Each line above is either **1 point** (meets all criteria) or **0** (otherwise). **Minimum deduction unit is 1 point; no 0.5-point deductions.**

---

### Convert .py to .ipynb using
```bash
jupytext --to ipynb choice-models.py
```

### Convert .ipynb to .py using
```bash
jupytext --to py choice-models.ipynb
```