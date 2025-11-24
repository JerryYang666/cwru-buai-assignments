# **A#5: Prompt Tuning & Response Control (LLM, 9 points)**

## **Assignment Instructions & Grading Rubric**

---

## **1) Purpose**

In this assignment, you will practice controlling and interpreting LLM outputs by changing key decoding and penalty parameters. Specifically, you will:

* Vary temperature and top_p to study creativity vs. stability in generation
* Adjust max_tokens to control the length of responses
* Use frequency_penalty to reduce repetition
* Use presence_penalty to encourage novelty and diversity
* Think about real-world marketing use cases where LLMs can be integrated into workflows

You must produce a reproducible Colab notebook (.ipynb).
The notebook is your single deliverable and should be submitted on Canvas.

---

## **2) Files & Data**

Starter notebook: **LLM_Assignment 5.ipynb** (provided on Canvas).

---

## **3) Tasks & Points — Total 9 points**

Each item below is graded as met (full credit) or not met (0). All questions assume that you follow the setup code provided in the starter notebook, including:

* Loading libraries
* Initialize the client API key

You will **not** use any external dataset for this assignment. You do **not** need to use HPC environment.

All questions assume you:

* Use the key and MODEL_PATH provided
* Follow the exact prompts and function names specified in the instructions
* Clearly label and print your outputs in the console
* Add Markdown reflections where requested

---

### **Q1 — Temperature & top_p Control (2 points)**

* (0.5 pt) Use the exact prompt, max_tokens = 150, and function name **show_slogan**
* (0.5 pt) Run 4 combinations of (temperature, top_p)
* (0.5 pt) Print all outputs clearly
* (0.5 pt) Markdown: comparison

---

### **Q2 — Output Length Control (2 points)**

* (0.5 pt) Use the exact email prompt, function name **show_email**, and temp/top_p settings
* (0.5 pt) Run 3 different max_tokens values (50, 500, 1000) with labels
* (0.5 pt) Print all outputs clearly
* (0.5 pt) Markdown: comparison

---

### **Q3 — Reducing Repetition with frequency_penalty (2 points)**

* (0.5 pt) Use exact description prompt + function name **show_description**, settings fixed
* (0.5 pt) Run 3 values of frequency_penalty (0.0, 0.5, 1.0) with labels
* (0.5 pt) Print all outputs clearly
* (0.5 pt) Markdown: comparison

---

### **Q4 — Encouraging Novelty with presence_penalty (2 points)**

* (0.5 pt) Use exact social-post prompt + function name **show_posts**, settings fixed
* (0.5 pt) Run 3 values of presence_penalty (0.0, 0.5, 1.0) with labels
* (0.5 pt) Print all outputs clearly
* (0.5 pt) Markdown: comparison

---

### **Q5 — LLM for Real Marketing Scenarios (1 point)**

* (0.5 pt) Propose new marketing use cases
* (0.5 pt) Choose one and describe prompts, expected outputs, workflow integration, and risks

---

## **4) Code & Library Rules**

Notebook must run top-to-bottom without errors (clean execution).
Do not re-import libraries already imported at the top of the notebook.

---

## **5) Submission Format**

File name: **LastName_FirstName_A5.ipynb**, submit on Canvas → Assignments → A#5.

---

## **6) Timeline & Late Submission**

Due date: See the Canvas A#5 item for the exact deadline.
Late policy: Up to 3 days late allowed. Minus 1 point per day late. Example: If 3 days late, the maximum score is 6 points, even with correct answers.

---

## **7) Academic Integrity & Collaboration**

Plagiarism or academic dishonesty is strictly prohibited. Discussion is encouraged, but you must write your own code and analysis. If you collaborated (idea-level discussion), list collaborators and the scope at the top of your notebook.

---

# **Grading Rubric (Checklist, 9 points total — no partial points)**

| Item     | Criterion (Yes = 1 / No = 0)                        | Notes |
| -------- | --------------------------------------------------- | ----- |
| **Q1-1** | Used exact prompt, meet other requirements (0.5 pt) |       |
| **Q1-2** | Ran all settings and labeled outputs (0.5 pt)       |       |
| **Q1-3** | Printed all outputs clearly (0.5 pt)                |       |
| **Q1-4** | Markdown comparison (0.5 pt)                        |       |
| **Q2-1** | Used exact prompt, meet other requirements (0.5 pt) |       |
| **Q2-2** | Ran all settings and labeled outputs (0.5 pt)       |       |
| **Q2-3** | Printed all outputs clearly (0.5 pt)                |       |
| **Q2-4** | Markdown comparison (0.5 pt)                        |       |
| **Q3-1** | Used exact prompt, meet other requirements (0.5 pt) |       |
| **Q3-2** | Ran all settings and labeled outputs (0.5 pt)       |       |
| **Q3-3** | Printed all outputs clearly (0.5 pt)                |       |
| **Q3-4** | Markdown comparison (0.5 pt)                        |       |
| **Q4-1** | Used exact prompt, meet other requirements (0.5 pt) |       |
| **Q4-2** | Ran all settings and labeled outputs (0.5 pt)       |       |
| **Q4-3** | Printed all outputs clearly (0.5 pt)                |       |
| **Q4-4** | Markdown comparison (0.5 pt)                        |       |
| **Q5-1** | Proposed 3–4 new marketing use cases                |       |
| **Q5-2** | Detailed workflow, prompts, outputs, risks          |       |

---

### Convert .py to .ipynb using
```bash
jupytext --to ipynb llm.py
```

### Convert .ipynb to .py using
```bash
jupytext --to py llm.ipynb
```