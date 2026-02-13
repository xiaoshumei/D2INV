import os

from openai import OpenAI
import tiktoken

import requests
import json


class LLM:
    def __init__(
        self,
        api_url="https://api.moonshot.cn/v1",
        model="kimi-k2-0905-preview",
        llm_vendor="kimi",
        content_length=256,
    ):
        assert llm_vendor in ["kimi", "deepseek", "openai", "vllm"]
        self.llm_vendor = llm_vendor
        self.api_url = api_url
        if llm_vendor == "kimi":
            base_url = "https://api.moonshot.cn/v1"
            self.model = "kimi-k2-0905-preview"
            self.max_tokens = 256 * 1024
        elif llm_vendor == "deepseek":
            base_url = "https://api.deepseek.com"
            self.model = "deepseek-chat"
            self.max_tokens = 128 * 1024
        elif llm_vendor == "openai":
            base_url = ""
            self.model = "gpt-5.2"
            self.max_tokens = 400000
        else:
            assert api_url, "api url must be provided when llm_vendor is 'vllm'"
            base_url = api_url + "/v1"
            self.max_tokens = content_length * 1024
            assert model, "model must be provided when llm_vendor is 'vllm'"
            self.model = model

        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY") if llm_vendor != "vllm" else "",
            base_url=base_url,
        )

    def calc_tokens(self, text):
        headers = {"Content-Type": "application/json"}
        if self.llm_vendor == "kimi":
            api_url = "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('LLM_API_KEY')}",
            }
            payload = {
                "messages": [{"role": "user", "content": text}],
                "model": self.model,
            }
            try:
                response = requests.post(
                    api_url, headers=headers, data=json.dumps(payload)
                )
                response.raise_for_status()
                result = response.json()
                return int(result["data"]["total_tokens"])
            except requests.exceptions.RequestException as e:
                print(e)
                return None
        elif self.llm_vendor == "deepseek":
            from transformers import AutoTokenizer

            def calculate_text_tokens():
                tokenizer = AutoTokenizer.from_pretrained(
                    "./tokenizer", trust_remote_code=False, use_fast=True
                )
                tokens = tokenizer.encode(text, add_special_tokens=False)

                return len(tokens)

            return calculate_text_tokens()
        elif self.llm_vendor == "openai":
            enc = tiktoken.encoding_for_model("gpt-4o")
            return len(enc.encode(text))

        else:
            api_url = self.api_url + "/tokenize"
            payload = {
                "prompt": text,
                "add_special_tokens": True,
            }
            try:
                response = requests.post(
                    api_url, headers=headers, data=json.dumps(payload)
                )
                response.raise_for_status()
                result = response.json()
                return int(result["count"])
            except requests.exceptions.RequestException as e:
                print(e)
                return None
