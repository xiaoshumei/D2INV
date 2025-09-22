import os
import requests

from openai import OpenAI


class LLM:
    def __init__(self, reason=True):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url="https://api.deepseek.com",
            # base_url="https://api.moonshot.cn/v1",
        )
        self.model = "deepseek-reasoner" if reason else "deepseek-chat"
        # self.model = "kimi-k2-0905-preview"
        self.max_tokens = 64 * 1024


class RequestsLLM:
    def __init__(self):
        self.API_URL = "xxx"
        self.API_KEY = "xxx"

    def chat(self, messages: list):

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "Qwen3-30B-A3B-Instruct-2507",
            "messages": messages,
            "max_tokens": 64 * 1024,
        }

        response = requests.post(self.API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]

            return answer
        else:
            return Exception(response.text)
