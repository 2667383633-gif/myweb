import os
from generate_breast_lesion_prompt_bank import call_llm_chat

text = call_llm_chat(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    model=os.environ["LLM_MODEL"],
    system_prompt="",
    user_prompt="Generate 3 short English prompts for breast tumor in ultrasound. Output only JSON.",
    timeout=120,
)

print(text)