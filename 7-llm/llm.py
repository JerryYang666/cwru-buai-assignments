# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="25577984"
# # Prompt Tuning & Response Control — 9‑Point Homework
#
# **Dataset:** None  
# Name: Ruihuang Yang  
# NetID: rxy216  
# Date: 11/24/2025  

# %% [markdown] id="1649abe8"
# ## 0. API Set Up

# %% id="35ad9dfa"
# Load basic libraries
# Do NOT import these libraries again below
# Re-importing (writing inefficient code) will result in a 0.5 point deduction

from openai import OpenAI
from dotenv import load_dotenv
import os, json, datetime, textwrap


def _require_env_var(name: str) -> str:
    """Fetch required env var and fail fast with a helpful message."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is missing. "
            "Set it in your shell or `.env` file."
        )
    return value


# Load environment variables from .env if present
load_dotenv()

# %% id="bbcbd883"
XLAB_API_KEY = _require_env_var("XLAB_API_KEY")
LLM_BASE_URL = _require_env_var("LLM_BASE_URL")
MODEL_PATH = _require_env_var("LLM_MODEL_PATH")
client = OpenAI(
    api_key=XLAB_API_KEY,
    base_url=LLM_BASE_URL
)


# %% colab={"base_uri": "https://localhost:8080/"} id="933cf1a3" outputId="edff4bef-22c1-4cf4-fea0-10b887f7f8b0"
#DO NOT RUN! This is an example code

def show_tweet_by_temperature(temp):
    resp = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content":  "Write a poetic tweet about AI and humanity discovering each other for the first time. /no_think"}],
        temperature=temp,
        max_tokens=2000
    )
    tweet = resp.choices[0].message.content.strip()
    print(f"\n Temperature = {temp}")
    print(tweet)

# Run demo
for t in [0.1, 0.9]:
    show_tweet_by_temperature(t)

# %% [markdown] id="8ae18372"
# ## Q1 (2 pt) — Temperature Controls
#
# **Goal:** Observe how model behavior changes when varying the sampling parameters `temperature` and `top_p`, while keeping the **prompt fixed**.
#
#
# ### **Tasks**
#
# 1. **Keep the prompt, max_tokens, and function name exactly as specified.** (0.5 pt)  
#    To receive full credit for this part, your code must:
#    - Use **exactly** the following prompt (no changes in wording or punctuation):  
#      `"Write five different short, catchy marketing slogans for a new coffee machine. Do not explain or think — only output the slogan itself."`
#    - Set **`max_tokens = 150`**
#    - Name the function **exactly** `show_slogan`
#
# 2. **Run the model with four different sampling settings** on the **same prompt.** (0.5 pt)
#
#    You will explore how `temperature` and `top_p` jointly influence creativity, tone, and originality in marketing messages.
#
#    - **`temperature`** controls **randomness**:  
#      - Lower values (e.g., `0.1`) → more focused, consistent, and “safe” slogans.  
#      - Higher values (e.g., `0.9`) → more varied, bold, and creative slogans.
#
#    - **`top_p` (nucleus sampling)** controls **how much of the probability mass** the model samples from:  
#      - Smaller values (e.g., `0.3`) → restricts the model to only the most likely word choices (predictable).  
#      - Larger values (e.g., `1.0`) → allows it to explore a broader set of possible words (diverse and imaginative).
#
#    **Run all four combinations below using the same prompt:**
#
#    | Setting | temperature | top_p |
#    |----------|--------------|-------|
#    | A | 0.1 | 0.3 |
#    | B | 0.1 | 1.0 |
#    | C | 0.9 | 0.3 |
#    | D | 0.9 | 1.0 |
#
#    For each case, clearly label and print the output (e.g., `= Setting A: temp=0.1, top_p=0.3 =`).
#
# 3. **Show all outputs** clearly labeled in the console. (0.5 pt)
#
# 4. **(Markdown cell)**  
#    Write a short comparison (3–4 bullet points) describing how the outputs differ. (0.5 pt)  
#    You may comment on:  
#    - Simplicity vs. creativity  
#    - Variety of slogans
#
# *Note:* In some cases, you may notice **no difference** among the four outputs.  
#    If no difference appears, simply write that there was no noticeable variation.  
#    If differences exist, describe how they change.  
#    Finally, **imagine you are a marketing manager**, explain **which sampling style** (e.g., low or high temperature)  
#    you would prefer to use for generating slogans, and why.

# %% [markdown] id="c679a9a8"
#

# %% [markdown] id="ddbeb455"
# ## Q2 (2 pt) — Output Length Control (max_tokens)
#
# **Goal:** Observe how changing max_tokens influences the generated content.
#
#
# ### **Tasks**
#
# 1. **Keep the prompt, temperature, top_p, and function name exactly as specified.** (0.5 pt)  
#    To receive full credit for this part, your code must:
#    - Use **exactly** the following prompt (no changes in wording or punctuation):  
#      `"Write a full email introducing a new coffee subscription service. Do not explain or think — only output the email itself."`
#    - Name the function **exactly** `show_email`
#    - temperature=0.5, top_p=0.5.
#
# 2. **Run the model three times, once for each max_tokens value: 50, 500, 1000** (0.5 pt)
#    For each case, clearly label and print the output (e.g., max_tokens = 50).
#
# 3. **Show all outputs** clearly labeled in the console. (0.5 pt)
#
# 4. **(Markdown cell)**  
#    Write a short comparison (3–4 bullet points) describing how the outputs differ. (0.5 pt)  

# %% [markdown] id="426da8c5"
#

# %% [markdown] id="1a120993"
# ## Q3 (2 pt) — Reducing Repetition (frequency_penalty)
#
# **Goal:** Observe how adjusting frequency_penalty changes repetition and naturalness in generated descriptions.
#
#
# ### **Tasks**
#
# 1. **Keep the prompt, temperature, top_p, max_token and function name exactly as specified.** (0.5 pt)  
#    To receive full credit for this part, your code must:
#    - Use **exactly** the following prompt (no changes in wording or punctuation):  
#      `"Write a detailed product description for a new coffee machine. Do not explain or think — only output the description itself."`
#    - Name the function **exactly** `show_description`
#    - temperature=0.7, top_p=0.9.
#    - max_token = 1000.
#
# 2. **Run the model three times, once for each frequency_penalty value: 0.0, 0.5, 1.0** (0.5 pt)
#    frequency_penalty is a parameter that reduces the likelihood of the model repeating the same words or phrases in its output.
#    For each case, clearly label and print the output (e.g., frequency_penalty = 0.5).
#
# 3. **Show all outputs** clearly labeled in the console. (0.5 pt)
#
# 4. **(Markdown cell)**  
#    Write a short comparison (3–4 bullet points) describing how the outputs differ. (0.5 pt)  

# %% [markdown] id="16f89e27"
#

# %% [markdown] id="50b7c8a8"
# ## Q4 (2 pt) — Encouraging Novelty (presence_penalty)
#
# **Goal:** Observe how presence_penalty encourages the model to generate more diverse or exploratory ideas.
#
#
# ### **Tasks**
#
# 1. **Keep the prompt, temperature, top_p, max_token and function name exactly as specified.** (0.5 pt)  
#    To receive full credit for this part, your code must:
#    - Use **exactly** the following prompt (no changes in wording or punctuation):  
#      `"Generate five creative social media post ideas to promote a new coffee machine. Do not explain or think — only output the post itself."`
#    - Name the function **exactly** `show_posts`
#    - temperature=0.8, top_p=0.9.
#    - max_token = 1000.
#
# 2. **Run the model three times, once for each presence_penalty value: 0.0, 0.5, 1.0** (0.5 pt)
#    presence_penalty is a parameter that encourages the model to introduce new ideas or words instead of repeating ones it has already used.
#    For each case, clearly label and print the output (e.g., presence_penalty = 0.5).
#
# 3. **Show all outputs** clearly labeled in the console. (0.5 pt)
#
# 4. **(Markdown cell)**  
#    Write a short comparison (3–4 bullet points) describing how the outputs differ. (0.5 pt)  

# %% id="2b423ee5"

# %% [markdown] id="c3d09372"
# ## Q5 (1 pt) — Applying LLMs to Real Marketing Scenarios
#
# Think creatively about how LLMs can be applied to solve real marketing problems.
#
# So far, you have experimented with LLMs to:
#
# Generate marketing slogans, Write promotional emails, Craft product descriptions, Suggest social media post ideas (Q4)
#
# These tasks show how LLMs can automate and enhance content creation.
# Now, imagine you are a digital marketing strategist planning to integrate LLM tools into your marketing workflow.
#
# **Tasks**
#
# 1. Propose 3 new marketing use cases for LLMs (beyond the four above).
#
# - Think beyond copywriting.
# - Your ideas can involve areas such as: Market research or trend detection, Customer segmentation and personalization, Sentiment or review analysis, A/B test generation, etc.
#
# 2. Provide specific task or workflow step for one marketing use case.
#
# - Choose one of your ideas and describe concretely how you would implement it using an LLM.
# - Describe the input prompts you might use.
# - Explain what outputs you would expect.
# - Briefly discuss potential challenges or risks (e.g., hallucination, bias).
#
# **Submission Format**
#
# - Markdown cell only — no code required.
# - Write your response in 2 sections:
#
# Section A: List 3–4 new marketing use cases (bullet points)
#
# Section B: Choose one idea and describe:

# %% id="a375f9ad"
