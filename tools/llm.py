from openai import OpenAI


def get_llm(vendor="openai", model_type="general"):
    client = OpenAI(
    api_key="xxx",
    base_url="xxx",
    )
    model = "gpt-4o" if model_type == "general" else "gpt-o1"
    
    return [client, model]
