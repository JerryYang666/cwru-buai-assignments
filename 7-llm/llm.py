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
import os
import textwrap


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

# %% id="ff39df6b"
SLOGAN_PROMPT = (
    "Write five different short, catchy marketing slogans for a new coffee machine. "
    "Do not explain or think — only output the slogan itself. /no_think"
)


def show_slogan(temperature: float, top_p: float) -> str:
    """Generate slogans with the exact settings required in Q1."""
    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": SLOGAN_PROMPT}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


SLOGAN_SETTINGS = [
    ("A", 0.1, 0.3),
    ("B", 0.1, 1.0),
    ("C", 0.9, 0.3),
    ("D", 0.9, 1.0),
]

for label, temp, nucleus in SLOGAN_SETTINGS:
    print(f"\n= Setting {label}: temp={temp}, top_p={nucleus} =")
    slogans = show_slogan(temp, nucleus)
    print(textwrap.indent(slogans, "  "))

# %% [markdown] id="c679a9a8"
# **Q1 Observations**
#
# - Settings A and C returned the exact same five slogans, showing that temperature alone didn’t move the needle when `top_p` stayed tight.  
# - Only Setting D (0.9 / 1.0) produced minor variation—mainly bold formatting—while the slogan wording still overlapped heavily with A–C.  
# - The `/no_think` prompt plus deterministic sampling likely constrained creativity more than expected.  
# - To surface new ideas I’d now adjust the prompt or add randomness elsewhere; otherwise the low-temperature run is sufficient for polished copy.

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

# %% id="08bb6aab"
EMAIL_PROMPT = (
    "Write a full email introducing a new coffee subscription service. "
    "Do not explain or think — only output the email itself. /no_think"
)


def show_email(max_tokens: int) -> str:
    """Generate the required email while varying only the token budget."""
    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": EMAIL_PROMPT}],
        temperature=0.5,
        top_p=0.5,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


for token_budget in (50, 500, 1000):
    print(f"\n= Email run: max_tokens={token_budget} =")
    email_text = show_email(token_budget)
    print(textwrap.indent(email_text, "  "))

# %% [markdown] id="426da8c5"
# **Q2 Observations**
#
# - `max_tokens=50` stopped after the headline and opening sentence—barely enough context for an email.  
# - `max_tokens=500` delivered a complete message with feature bullets, CTA, and branded sign-off.  
# - `max_tokens=1000` added personalization placeholders and extra closing details but didn’t meaningfully expand content beyond the 500-token draft.  
# - A cap around 350–500 tokens still feels ideal: roomy enough for structure without fluff.

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

# %% id="d82dc5f8"
DESCRIPTION_PROMPT = (
    "Write a detailed product description for a new coffee machine. "
    "Do not explain or think — only output the description itself. /no_think"
)


def show_description(frequency_penalty: float) -> str:
    """Generate the machine description while sweeping frequency_penalty."""
    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": DESCRIPTION_PROMPT}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000,
        frequency_penalty=frequency_penalty,
    )
    return response.choices[0].message.content.strip()


for freq_penalty in (0.0, 0.5, 1.0):
    print(f"\n= Description run: frequency_penalty={freq_penalty} =")
    description = show_description(freq_penalty)
    print(textwrap.indent(description, "  "))

# %% [markdown] id="16f89e27"
# **Q3 Observations**
#
# - With `frequency_penalty=0.0` the copy leaned hard on repeated adjectives like “rich,” “luxury,” and “barista-grade.”  
# - At 0.5 the wording diversified while keeping a natural storyline—ideal for product detail pages.  
# - With the heavy penalty (1.0) the model avoided repeats so aggressively that it occasionally inserted quirky synonyms that felt off-brand.  
# - Sweet spot: 0.3–0.6 keeps prose fresh without sacrificing tonal consistency.

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
POST_PROMPT = (
    "Generate five creative social media post ideas to promote a new coffee machine. "
    "Do not explain or think — only output the post itself. /no_think"
)


def show_posts(presence_penalty: float) -> str:
    """Generate social post ideas for Q4."""
    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": POST_PROMPT}],
        temperature=0.8,
        top_p=0.9,
        max_tokens=1000,
        presence_penalty=presence_penalty,
    )
    return response.choices[0].message.content.strip()


for presence_penalty in (0.0, 0.5, 1.0):
    print(f"\n= Social post run: presence_penalty={presence_penalty} =")
    posts = show_posts(presence_penalty)
    print(textwrap.indent(posts, "  "))

# %% [markdown] id="c3d09372"
# **Q4 Observations**
#
# - All three runs leaned on similar “wake up to perfection” narratives, so presence_penalty had limited impact with this tightly phrased prompt.  
# - The 0.5 setting introduced slight variety (UGC callouts, “coffee alchemist” phrasing) without drifting off brief.  
# - Even at 1.0 the ideas stayed familiar, implying we may need additional creative constraints or different prompts for true novelty.  
# - I’d pair presence_penalty tweaks with more open-ended instructions when the goal is wildly different social angles.
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

# %% [markdown] id="a375f9ad"
# ### Q5 — LLMs for Real Marketing Scenarios
#
# **Section A — New Use Cases**
# - Competitive intelligence digests: summarize weekly feature launches, promo angles, and pricing moves from competitor sites, blogs, and filings.  
# - Audience micro-segmentation insights: analyze CRM notes + survey responses to surface nuanced personas and messaging cues.  
# - Experiment backlog generator: transform KPI deltas and creative briefs into prioritized A/B test concepts with hypotheses and metrics.  
# - CX escalation triage: read long-form support transcripts and draft personalized apology + retention plans for at-risk customers.
#
# **Section B — Deep Dive (Experiment backlog generator)**
# - **Workflow:** After each campaign retro, export KPIs (open/click/conversion lift), notable creative elements, and audience splits. Feed them to an LLM prompt that asks for net-new A/B tests ranked by potential impact.  
# - **Prompt sketch:** “You are a lifecycle marketing lead. Using the metrics + context below, propose A/B tests to improve conversion. For each: hypothesis, proposed change, success metric, rollout risk. Data: {table/json dump}.”  
# - **Expected outputs:** Structured list or table with clear hypotheses (“Switch hero image to focus on pour-over ritual to increase CTR”), concrete asset tweaks, and measurement guidance.  
# - **Risks:** If the data is noisy the model may hallucinate causality or suggest tests we already ran, so I’ll keep human review, link every hypothesis back to provided metrics, and note uncertainty when inputs are incomplete.