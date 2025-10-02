# Assignment 3 – Tokenization & Lemmatization

## A#3: Tokenization & Lemmatization (Amazon, 9 points)

## 1) Purpose

For each problem, perform the tokenization or lemmatization as required. If the question asks for an explanation, provide a clear explanation in a separate text cell within your notebook.

You must produce a **reproducible Colab notebook (.ipynb)**. The notebook is your single deliverable and should be submitted on Canvas.

---

## 2) Files & Data

* **Starter notebook**: `Token_Lemma_Assignment_3.ipynb` (provided on Canvas)
* **Reference notebooks**: `S4_Tokenization.ipynb` + `S5_Lemmatization.ipynb`
* **Dataset**: `Amazon Musical.xlsx`
* **Target column**: `review_body`

---

## 3) Tasks & Points — Total 9 points

*Minimum deduction unit is 1 point. There are no 0.5-point deductions. Each item below is graded as met (full credit) or not met (0).*

### Q1 — Keras Tokenizer (1 point)

* Import the function `text_to_word_sequence` from `tensorflow.keras.preprocessing.text`
* Convert the `review_body` column to string type to avoid errors when non-string values appear: use `.astype(str)`
* Apply `text_to_word_sequence` to each row of `review_body` and store the result in a new column named `token_keras`
* Preview the result by printing the original `review_body` and the new `token_keras` column for the first 5 rows

---

### Q2 — Regex Tokenizer Version 1 (1 point)

* Import the built-in `re` module for regular expressions
* Define a compiled regex pattern: `r"[A-Za-z]+"`

  * Matches one or more English letters (A–Z or a–z)
  * Numbers, punctuation, emojis, and symbols are excluded
  * Hyphenated words (e.g., cost-effective) will be split (cost, effective)
  * No lowercasing or other normalization performed here
* Ensure `review_body` is treated as a string: `.astype(str)`
* Apply `pattern.findall(x)` to each row of `review_body` with `.apply(...)`
* Save the result to a new column named `token_regex_ver1`
* Preview by printing `review_body` and `token_regex_ver1` for the first 5 rows

---

### Q3 — Regex Tokenizer Version 2 (1 point)

* Define a compiled regex pattern: `r"\w+"`, name the object `pattern2`

  * Matches one or more “word characters” (letters A–Z/a–z, digits 0–9, underscore `_`)
  * Punctuation, emojis, and symbols are excluded
  * Hyphenated words will be split
  * No lowercasing or normalization performed here
* Ensure `review_body` is treated as string: `.astype(str)`
* Apply `pattern2.findall(x)` to each row of `review_body` with `.apply(...)`
* Save the result to a new column named `token_regex_ver2`
* Preview by printing `review_body` and `token_regex_ver2` for the first 5 rows

---

### Q4 — Regex Tokenizer Version 3 & Removing Some Stop Words (2 points)

* Define a stoplist as a Python set named **STOPLIST**:
  `{"the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "br"}`
* Define a regex pattern object named `pattern3`:
  `pattern3 = re.compile(r"[A-Za-z_']+")`

  * Matches sequences of letters, underscores, or apostrophes
  * Allows simple handling of contractions like *don’t*
  * Numbers and punctuation (other than `'` and `_`) excluded
* Ensure `review_body` is treated as string: `.astype(str)`
* Convert `review_body` to lowercase: `.str.lower()`
* Extract tokens with `pattern3.findall(x)`
* Remove any tokens found in the stoplist (`if w not in STOPLIST`)
* Save results to a new column: `token_regex_ver3`
* Preview by printing both `review_body` and `token_norm` for the first 5 rows

---

### Q5 — Lemmatizer for Regex Version 2 (2 points)

* Import required libraries: `spaCy` and `nltk`
* Download the NLTK stopwords list: `nltk.download('stopwords')`
* Load English stopwords from NLTK into a Python set `stop_words`
* Load the spaCy English model: `en_core_web_sm` → variable `nlp`
* Define function **`lemmatize_tokens`** that:

  1. Takes a list of tokens as input
  2. Joins them into a string for spaCy to process
  3. Creates a spaCy `doc` object
  4. Extracts each token’s lemma (base form)
  5. Converts lemma to lowercase
  6. Keeps only alphabetic tokens (`token.is_alpha`)
  7. Removes tokens found in `stop_words`
* Apply `lemmatize_tokens` to the column `token_regex_ver2` to produce a new column `lemmas`
* Preview by printing `review_body` and `lemmas` for the first 5 rows

*Requirement*: Follow the steps in `S5_Lemmatization.ipynb`. The only change is to apply the process to `token_regex_ver2` instead of `review_body`.

---

### Q6 — Suggestion of Your Own Tokenizer (2 points)

* Create what you believe is the **most appropriate tokenization method** for this dataset
* You may freely combine any techniques introduced above

  * Multiple regex patterns step by step
  * Decide whether or not to include stopword removal
  * Decide whether or not to apply lemmatization
* Think of this task as preparing Amazon review data for a real project
* Choose the method you find most suitable, implement it, and provide a **clear explanation of why** you selected that approach
* Rules:

  * Code must run without errors
  * Explanation must justify your choices
  * Do **not** simply reuse one of the five previous tokenizers without modification
  * Output may end up looking the same as a previous method (depending on data), and that is fine — what matters is your logic is different

---

## 4) Code & Library Rules

* Notebook must run top-to-bottom without errors (clean execution).

---

## 5) Submission Format

* File name: `LastName_FirstName_A3_Tokenization.ipynb`
* Add a markdown header at the top with: your name, student ID, course, and assignment title
* Submit on Canvas → Assignments → A#3

---

## 6) Timeline & Late Submission

* **Due date**: See Canvas A#3 item
* **Late policy**: Up to 3 days late allowed. Minus 1 point per day late.

  * Example: 3 days late → maximum score is 6 points (even if correct)

---

## 7) Academic Integrity & Collaboration

* Plagiarism or academic dishonesty is strictly prohibited
* Discussion is encouraged, but you must write your own code and analysis
* If you collaborated (idea-level discussion), list collaborators and the scope at the top of your notebook

---

## Grading Rubric (Checklist, 9 points total — no partial points)

**Scoring principle**: Each line is worth 1 point (yes = 1, no = 0). Minimum deduction = 1 point; no 0.5-point deductions.

| Item | Criterion                                                                                                 | Notes |
| ---- | --------------------------------------------------------------------------------------------------------- | ----- |
| Q1   | Student correctly applies `text_to_word_sequence` on `review_body` and creates `token_keras` column       |       |
| Q2   | Student correctly defines regex `[A-Za-z]+`, applies it to `review_body`, and creates `token_regex_ver1`  |       |
| Q3   | Student correctly defines regex `\w+`, applies it to `review_body`, and creates `token_regex_ver2`        |       |
| Q4-1 | Student correctly defines stoplist and regex `[A-Za-z_']+`, applies lowercase conversion and tokenization |       |
| Q4-2 | Student removes stopwords and saves results in `token_regex_ver3`                                         |       |
| Q5-1 | Student loads spaCy + NLTK stopwords, defines `lemmatize_tokens` function properly                        |       |
| Q5-2 | Student applies function to `token_regex_ver2` and creates `lemmas` column                                |       |
| Q6-1 | Student implements a tokenizer different from the five methods above (combination/modified)               |       |
| Q6-2 | Student provides a clear explanation of why their approach is appropriate                                 |       |

---

### Convert .py to .ipynb using
```bash
jupytext --to ipynb tokenization-lemmatization.py
```

### Convert .ipynb to .py using
```bash
jupytext --to py tokenization-lemmatization.ipynb
```