# Assignment – Topic Modeling

# A#4: TF-IDF & Topic Modeling (Amazon, 9 points)

---

## 1) Purpose

In this assignment, you will practice three key steps in text analytics:

* Building a **TF-IDF representation with up to tri-grams**
* Applying **MiniBatchNMF** for topic modeling under different hyperparameter settings
* Interpreting how modeling choices (**number of topics, initialization, batch size**) affect topic quality

You must produce a **reproducible Colab notebook (.ipynb)**.
The notebook is your **single deliverable** and should be submitted on **Canvas**.

---

## 2) Files & Data

* **Starter notebook**: `TF_IDF_Topic_Assignment_4.ipynb` (provided on Canvas).
* **Dataset**: `Amazon Musical.csv`.
* **Target column**: `review_body`

---

## 3) Tasks & Points — Total **9 points**

Each item below is graded as **met (full credit)** or **not met (0)**.

All questions assume that you follow the **setup code** provided in the starter notebook, including:

* Loading libraries
* Loading `Amazon Musical.csv` into a DataFrame `df`
* Running the provided `spacy_analyze_pipe` function to create `df["spacy_tokens"]`
* Creating `df["lemmas"]` and `df["lemmas_text"]` as shown in the starter code

> You must **not re-import libraries** that are already imported at the top of the notebook.
> If your code repeatedly imports libraries in this section and shows inefficiency, **1 point will be deducted separately** from the other criteria.

---

### Q1 — TF-IDF with up to 3-grams (**1 point**)

*  Build a **tri-gram TF-IDF matrix** (**0.5**)
*  **Inspect the top-scoring terms** (**0.5**)

>  If you change variable names or TF-IDF parameters, or do not use `get_top_terms` exactly as instructed, you will lose the point.

---

###  Q2 — Very coarse topics (K = 2 baseline) (**1 point**)

*  Set up a **MiniBatchNMF model** with:

  * `n_components = 2` (**K = 2 topics**)
  * `init = "nndsvda"`
  * `random_state = 42`
  * `max_iter = 300`
  * `batch_size = 512`
*  Fit the model on `X` (**`X_list_123g`**)
*  Print the **shapes of W and H**
*  For each topic, print the **top 10 terms** using
  `vocab = vec_list_trigram.get_feature_names_out()`
*  In a **Markdown cell**: For each topic, look at the keywords and **create your own topic name**

---

###  Q3 — Increase topic count (K = 4) (**1 point**)

*  Set up a **MiniBatchNMF model**:

  * Starting from your Q1 code, change the number of topics to **K = 4**
    (`n_components = 4`)
  * Keep all other parameters the same
*  Fit the model on `X` (`X_list_123g`)
*  Print the **shapes of W and H**
*  For each topic, print the **top 10 terms** using
  `vocab = vec_list_trigram.get_feature_names_out()`
*  In a **Markdown cell**: For each topic, look at the keywords and **create your own topic name**

---

###  Q4 — Same K = 4 but different initialization (**1 point**)

*  Set up a **MiniBatchNMF model**:

  * Change the initialization method from `"nndsvda"` to `"random"`
  * Keep all other parameters the same
*  Fit the model on `X` (`X_list_123g`)
*  Print the **shapes of W and H**
*  For each topic, print the **top 10 terms** using
  `vocab = vec_list_trigram.get_feature_names_out()`
*  In a **Markdown cell**: For each topic, look at the keywords and **create your own topic name**

---

###  Q5 — Same K = 4, add early stopping rule (**1 point**)

*  Set up a **MiniBatchNMF model**
*  Print the **shapes of W and H**
*  For each topic, print the **top 10 terms** using
  `vocab = vec_list_trigram.get_feature_names_out()`
*  In a **Markdown cell**: For each topic, look at the keywords and **create your own topic name**

---

###  Q6 — Same K = 4, change batch size & iterations (**1 point**)

*  Set up a **MiniBatchNMF model**
*  Print the **shapes of W and H**
*  For each topic, print the **top 10 terms** using
  `vocab = vec_list_trigram.get_feature_names_out()`
*  In a **Markdown cell**: For each topic, look at the keywords and **create your own topic name**

---

###  Q7 — Reflection on topic modeling choices (**3 points**)

Write a **short reflection in a Markdown cell** addressing the 5 questions below:

1.  **Effect of the number of topics** (**0.5 pt**):
2.  **Effect of initialization** (**0.5 pt**):
3.  **Effect of early stopping** (**0.5 pt**):
4.  **Effect of batch size and iterations** (**0.5 pt**):
5.  **Summarize which configuration you would choose** as your **“final” topic model** for this dataset and **briefly justify your choice** (**1 pt**)

---

## 4) Code & Library Rules

* Notebook must run **top‑to‑bottom without errors** (clean execution)

---

## 5) Submission Format

* File name: `LastName_FirstName_A4.ipynb`
* Submit on Canvas → **Assignments → A#4**

---

## 6) Timeline & Late Submission

* **Due date**: See the Canvas A#4 item for the exact deadline.
* **Late policy**: Up to **3 days late** allowed.
  Minus **1 point per day late**.

> Example: If 3 days late, the maximum score is **6 points**, even with correct answers.

---

## 7) Academic Integrity & Collaboration

* **Plagiarism or academic dishonesty is strictly prohibited**.
* **Discussion is encouraged**, but you must **write your own code and analysis**.
* If you **collaborated (idea‑level discussion)**, list collaborators and the scope **at the top of your notebook**.

---

## Grading Rubric (Checklist — 9 points total, **no partial points**)

| Item | Criterion                                  | Points |
| ---- | ------------------------------------------ | ------ |
| Q1-1 | Build the TF-IDF vectorizer                | 0.5    |
| Q1-2 | Display top trigrams                       | 0.5    |
| Q2   | Implemented code + topic naming (Markdown) | 1      |
| Q3   | Implemented code + topic naming (Markdown) | 1      |
| Q4   | Implemented code + topic naming (Markdown) | 1      |
| Q5   | Implemented code + topic naming (Markdown) | 1      |
| Q6   | Implemented code + topic naming (Markdown) | 1      |
| Q7-1 | Effect of number of topics                 | 0.5    |
| Q7-2 | Effect of initialization                   | 0.5    |
| Q7-3 | Effect of early stopping                   | 0.5    |
| Q7-4 | Effect of batch size and iterations        | 0.5    |
| Q7-5 | Final configuration and justification      | 1      |

---

### Convert .py to .ipynb using
```bash
jupytext --to ipynb topic-modeling.py
```

### Convert .ipynb to .py using
```bash
jupytext --to py topic-modeling.ipynb
```