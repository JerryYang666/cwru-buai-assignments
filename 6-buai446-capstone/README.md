# **Capstone Project Description: Recommender System Based on Online Reviews**

**Submission Deadline: Dec 2, 2025**

In lieu of the final exam, students will work on a capstone project. This project will span the semester, allowing students to apply the skills learned across various courses in the program—as well as those they acquire independently—to solve a real-world problem. The project will be **data-driven**, focusing on the development of a **recommender system based on online reviews** that include both text and customer ratings.

---

## **Project Objectives**

Students are expected to demonstrate their proficiency in the following key areas:

### **1. Problem Formulation**

* Clearly define the decision problem that can be addressed using the provided dataset.
* The problem formulation should specify:

  * **Business context**
  * **Objectives**
  * **The role of the recommender system in supporting decision-making**
* Students should articulate the problem with confidence and ensure feasibility given the available data.

---

### **2. Data Wrangling**

The provided data will require significant preprocessing before it is suitable for analysis.

This includes:

* **Data Cleaning:** Handling missing values, outliers, inconsistencies
* **Transformation:** Text processing, feature engineering
* **Rescaling:** Applying normalization when appropriate

Students should **document all steps** taken to prepare the data and justify their decisions according to data science best practices.

---

### **3. Analytics Plan**

Students should develop a comprehensive analytics plan outlining the approach to building the recommender system, including:

* **Methodologies:**
  Examples: collaborative filtering, content-based filtering
* **Model Evaluation:**
  Metrics such as RMSE, MAE, precision, recall
* **Model Selection Criteria**
* **Iterative Workflow:**
  Train–test–validate cycles and model comparison to ensure robustness and validity.

---

### **4. Insights for Decision-Making**

After executing the analytics plan, students must translate the results into business insights:

* Interpret findings in **clear, non-technical language**
* Provide **actionable recommendations**
* Demonstrate how the recommender system can support decision-making or improve customer satisfaction

---

### **5. (Bonus) AI Personalized Automation**

(Optional advanced challenge)

Students may develop an **AI chatbot** capable of:

* Personalizing recommendations for individual users
* Integrating NLP and user profiling
* Providing dynamic interaction and user-specific recommendations

---

### **6. (Bonus) Business Application Analysis**

(Optional advanced challenge)

Students may analyze how leading companies (e.g., Amazon, Spotify, Netflix) implement recommender systems.

Analysis should include:

1. **Selection of Two Companies**

   * Provide an overview and business model for each.

2. **Recommender System Analysis for Each Company**

   * **Data Collection:** Browsing behavior, ratings, purchase history, search queries
   * **Features and Metrics** used in recommendations
   * **Algorithms:** Collaborative filtering, content-based, hybrid, deep learning
   * **System Evolution:** How it has changed over time
   * **Strategic Alignment:** How recommendations support company strategy
   * **Ethical Considerations:** Privacy, bias, transparency

3. **Comparative Analysis**

   * Effectiveness or innovation of each approach
   * Advantages and disadvantages
   * If from the same industry, compare strategic differences

**Student Creativity:**
Students are encouraged to integrate outside readings, interviews, or their own observations.

---

# **Course/Assignment Breakdown**

---

## **Part 1: Problem Formulation and Data Preparation**

### **Objectives**

### **Problem Formulation**

**Template Requirements:**

* **Business Context:** Describe the industry and organizational setting
* **Objectives:** Specify what the recommender system aims to achieve

  * Example: *Amazon Music Recommendation*
* **Decision-Making Role:** Explain how the system supports decision-making

**Student Creativity:**
Students may choose any business context and specify unique objectives based on the dataset.

---

### **Data Wrangling**

**Documentation Requirements:**

* **Data Cleaning:** Methods for handling missing values, outliers, inconsistencies
* **Data Transformation:** Text processing, feature engineering
* **Rescaling/Normalization:** Justify any scaling methods used

**Student Creativity:**
Students may select innovative preprocessing techniques and justify their choices.

---

## **Part 2: Analytics Implementation and Insights**

### **Objectives**

### **Analytics Plan and Model Development**

**Sections to Include:**

* **Methodologies:** Algorithms such as collaborative filtering, content-based filtering
* **Evaluation Metrics:** RMSE, MAE, precision, recall
* **Model Selection Criteria**
* **Student Creativity:**

  * Experimentation with various algorithms
  * Justifying chosen approaches

---

### **Insights for Decision-Making**

**Report Template Requirements:**

* **Interpretation of Results:** Translate technical outputs into layman's terms
* **Business Recommendations:** Provide actionable suggestions for decision-makers

**Student Creativity:**
Students are encouraged to offer unique insights and recommendations.

---

### **(Bonus) AI Personalized Automation**

Optional challenge:
Build an AI chatbot that integrates with the recommender system.

**Includes:**

* NLP integration
* User profiling
* Real-time personalization

Creativity is fully encouraged.

---

### **(Bonus) Business Application Analysis**

Optional challenge: Analyze real-world companies’ recommender systems.

**Guidelines Recap:**

1. Choose **two companies**
2. Analyze their recommender systems across:

   * Data collection
   * Features and metrics
   * Algorithms used
   * Evolution over time
   * Strategic alignment
   * Ethical considerations
3. Provide comparative insights

Creativity encouraged through use of external sources.

---

# **Key Skills from NLP, Recommendation Engines, and Chatbot Development**

---

## **1. Natural Language Processing (NLP) Techniques for Text Analysis**

*(Covered in MBAI 435)*

### **Key Skills**

* **Text Preprocessing**

  * Tokenization
  * Stemming
  * Lemmatization
  * Stop-word removal
* **Feature Extraction**

  * TF-IDF
  * Word embeddings (Word2Vec, GloVe)
  * Transformer models (BERT)
* **Sentiment Analysis**
* **Topic Modeling** (LDA)

### **Why It’s Important**

NLP enables extraction of insights from unstructured text reviews, improving personalization and accuracy.

---

## **2. Recommender System Algorithms and Implementation**

### **Key Skills**

* **Collaborative Filtering**

  * User-based
  * Item-based
* **Content-Based Filtering**
* **Hybrid Systems**
* **Matrix Factorization (SVD, etc.)**

### **Why It’s Important**

These algorithms form the backbone of most recommendation engines, allowing tailored item suggestions based on user behavior and item attributes.

---

## **3. Chatbot Development and Integration with Recommender Systems**

*(Bonus — Student Initiative)*

### **Key Skills**

* **Conversational Design:** Dialogue flow, context-awareness
* **Chatbot Frameworks:** Rasa, Dialogflow, Microsoft Bot Framework
* **Personalization:** Using user data to tailor responses
* **API Integration:** Linking chatbot to recommender backend

### **Why It’s Important**

Chatbots offer interactive and personalized recommendation experiences for users.

---

## **4. Advanced NLP for Conversational AI**

*(Advanced & Optional; Bonus)*

### **Key Skills**

* **Natural Language Understanding (NLU):** Intent detection, entity extraction
* **Dialogue Management:** State tracking, multi-turn dialogue
* **Natural Language Generation (NLG):** Contextual and coherent response generation
* **Transformer Models:** BERT, GPT; fine-tuning for domain-specific tasks

### **Why It’s Important**

Advanced NLP enables sophisticated chatbots capable of deeper understanding and more helpful interaction.

---

# **Summary**

By focusing on these key areas—NLP, recommender systems, and chatbot development—students will be equipped to deliver a high-quality, end-to-end solution that meets all project objectives.

---

### Convert .py to .ipynb using
```bash
jupytext --to ipynb buai446-capstone.py
```

### Convert .ipynb to .py using
```bash
jupytext --to py buai446-capstone.ipynb
```