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
        messages=[{"role": "user", "content":  "Write a poetic tweet about AI and humanity discovering each other for the first time."}],
        temperature=temp,
        max_tokens=2000
    )
    tweet = resp.choices[0].message.content.strip()
    print(f"\n Temperature = {temp}")
    print(tweet)

# Run demo
for t in [0.1, 0.9]:
    show_tweet_by_temperature(t)