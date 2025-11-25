# **BUAI 435 — Final Project Guidelines (Fall 2025)**

*Last updated: November 2, 2025 (ET)*

---

## **1) Overview & Options**

Choose one of the two scenarios below:

### **Option 1 — Amazon Musical Instruments Sentiment Modeling**

**Data:** Amazon Musical.csv
The main goal of this project is to build a sentiment model using the Amazon Musical Instruments Review dataset. The primary outcomes are

1. A labeled dataset where each review is classified by sentiment, and
2. Actionable insights derived from sentiment patterns.

For the second outcome, your analysis must include some actionable insights derived from sentiment patterns, for instance:

* Understanding how customer satisfaction evolves over time, or
* Identifying which products or themes drives positive or negative sentiment.
* Any additional analysis that helps interpret customer satisfaction through sentiment (e.g., comparing products, comparing rating/sentiment distributions, or linking sentiment to review length) is also very encouraged.

Teams may optionally conduct topic modeling to deepen their analysis, for example, by:

* Calculating the proportion of positive and negative reviews within each topic to identify which themes are the core drivers of satisfaction or dissatisfaction, or
* Using topic probabilities as features to improve the performance of the sentiment classifier.

In other words, topic modeling can be used to enhance your sentiment analysis, but it is not a required component of this scenario. The core focus remains on conducting sentiment analysis as an input for a hypothetical recommender system. Of course, effective use of topic modeling will naturally be a factor that contributes to a higher score.

### **Option 2 — Tesla Community Topic Modeling & Sentiment Analysis**

**Data:** Tesla.csv
In this option, teams will use the Tesla User Community dataset to perform both sentiment analysis and topic modeling. Rather than building a recommender system, your goal is to take an executive-strategy perspective, analyzing how users express their opinions, concerns, and excitement about Tesla products and technologies, and then translating those insights into a five-year strategic plan for Tesla’s leadership.

The analysis must combine sentiment patterns with qualitative topic insights to provide a well-rounded view of user perceptions.

The primary outcomes are:

1. Sentiment labeling and evaluation
2. Topic modeling and interpretation (e.g., safety, charging infrastructure, pricing, performance)
3. Strategic plan – Based on your analytical findings, this plan can take any form of actionable proposal derived from your results.

For your third outcome: Your analysis must include actionable insights that integrate findings from both sentiment patterns and topic trends. For example:

* Trend analysis – Track how public sentiment within specific discussion topics evolves over time (e.g., spikes in discussion following software updates or major incidents).
* Comparative insight – Examine how different topics correspond to varying sentiment levels (e.g., discussions about performance may remain positive, while those about charging infrastructure show more negative sentiment).

---

## **2) Task Breakdown**

You can choose either Option 1 or Option 2 for this project.
Both options follow the same overall structure described below; however:

* **Option 1:** Topic Modeling (Task 4) and Sentiment & Topic Integration (Task 5) are optional.
* **Option 2:** Both Task 4 and Task 5 are required.

---

### **Task 1: Data Preprocessing**

In this stage, students will clean and prepare the dataset for analysis.
For example:

* Standardize text: lowercase conversion, punctuation and noise removal.
* Tokenize and normalize using lemmatization and stop-word removal.
* Handle duplicates and missing values.

**Outcomes:** Documentation of each cleaning step with before-and-after examples.

---

### **Task 2: Text Feature Engineering**

Convert the cleaned text into numerical representations for modeling.
For example:

* Use TF-IDF (or comparable methods) to represent review text numerically.

**Outcomes:** Summary of the feature space (shape, top terms, etc.).

---

### **Task 3: Sentiment Analysis**

This is the main component for both options.
You may use a lexicon-based model, an LLM-based classifier, a supervised-learning model, or an ensemble of these approaches.

If you choose supervised learning, you must manually label a subset of the data to serve as the ground-truth set for evaluation.

**Outcomes:**

* Sentiment-labeled dataset.
* Model-evaluation summary
* Interpretive findings

---

### **Task 4: Topic Modeling**

*Optional for Option 1, Required for Option 2*

This is the main component for option 2. You may use NMF or LLM.
Topic modeling can help interpret why certain sentiments appear or identify customer/product segments. You should interpret top keywords per topic and assign meaningful labels.

**Outcomes:** Topic-summary

---

### **Task 5: Actionable insights that integrate findings from both sentiment patterns and topics**

*Optional for Option 1, Required for Option 2*

For example:

* Analyze sentiment trends over time and across topics to identify product-satisfaction dynamics (option 1) or public perception dynamics (option 2).

**Outcomes:** Written analysis or visualization summarizing sentiment/topic-based insights.

---

## **3) Final Deliverables (Submit on Canvas)**

Submit both:

1. Code package
2. Report PDF (~15 pages, excluding references/appendix)
3. Team meeting notes (1 page).
4. Individual contribution statements (each member, 1 page).

Presentations have no point value but are required; peer voting will award +1 extra credit to the top-voted team (added to the course total).

---

## **4) Teamwork, Process, and Fairness**

Free-riding will be handled strictly and fairly.

Required at term end:

* Team meeting notes + attendance (concise but real).
* Individual contribution statements (what you did, and what your teammates did).

Data/work split (recommended):

* For large datasets, divide and conquer. Example: split reviews by a stable rule (e.g., hash(review_id) % team_size) or by deterministic seeded sampling. If you use an LLM for labeling, partition the labeling work across members and merge with a clear, reproducible script.

---

## **5) Key Dates**

**In-class presentations:** Tue, Dec 2 - Live feedback + peer vote collected.
**Final submission (code + report):** Tue, Dec 9 @ 11:59 PM
One week after presentations to incorporate feedback.

---

## **6) Report Template (~15 pages)**

**Executive Summary (≤1 page).**
Problem, data, key findings, actionable takeaways.

**Business Scenario.**

* Option 1: Sentiment as input to recommender
* Option 2: Strategy context for Tesla

**Data & Preprocessing.**
filters, cleaning, tokenization, POS/lemma choices, etc.

**Methods.**
Sentiment approach (lexicon, ML, or LLM), topic models, vectorizers, hyperparameters; why these choices?

**Results (Examples)**
— Sentiment. Metrics (accuracy, by class), error analysis (confusions).
— Topics. Top terms, exemplar texts, label each topic and interpret.

**Business Insights.**

* Option 1: How sentiment improves recommender input, Customers’ pain points and unmet needs, A/B test ideas, etc.
* Option 2: 5-Year Strategic Plan with 3–5 initiatives, risks, and suggestions.

**Limitations.**
Bias, labeling noise, representativeness, LLM risks/costs if used.

**References & Appendix.**

---

## **7) Presentation (Dec 2)**

**Format:** 20 minutes talk + 10 minutes Q&A.
**Content:** problem framing, method snapshot, 2–3 strongest figures/tables, key takeaways.
**Voting:** quick peer vote in class; top team gets +1 extra credit.
**Professor’s Feedback:** use it to refine before final submission.

---

## **8) LLM Policy (Optional Use)**

LLMs are optional. You can fully complete the project with the classical toolchain taught in class. If you use LLMs (xLab server), you disclose prompt lists.

---

## **9) Grading Rubric (Total = 25 points)**

### **A. Code (10 pts)**

1. **Reproducibility & Structure (2.0)** — clear, deterministic seeds, runnable scripts.
2. **Preprocessing & Documentation (2.0)** — thoughtful cleaning, tokenization, and rationale.
3. **Model Implementation & Results Integration (5.0)** — overall modeling quality, appropriate methods, robustness of results, and interpretability.
4. **Efficiency (1.0)** — appropriate metrics, diagnostic checks, resource management, and HPC usage.

### **B. Report (15 pts)**

1. **Clarity & Organization (2.0)** — logical flow, high-quality writing/figures.
2. **Business Framing (2.0)** — actional insights and concrete recommendations.
3. **Methods Explanation (4.0)** — justify choices; parameters and trade-offs.
4. **Results Quality (4.0)** — rigorous metrics, compelling visuals, topic validity.
5. **Limitations & Ethics (2.0)** — bias, data limits, LLM risks/costs if used.
6. **Team Contributions (1.0)** — credible division of labor.

**Presentations:** required but ungraded. Peer-voted 1st place = +1 extra credit to overall course points.

---

### Convert .py to .ipynb using
```bash
jupytext --to ipynb buai435-final.py
```

### Convert .ipynb to .py using
```bash
jupytext --to py buai435-final.ipynb
```