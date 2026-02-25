import os

from openai import OpenAI
import tiktoken

import requests
import json


class LLM:
    def __init__(
        self,
        llm_vendor="kimi",
        content_length=256,
    ):
        assert llm_vendor in ["kimi", "deepseek", "openai", "vllm"]
        self.llm_vendor = llm_vendor
        if llm_vendor == "kimi":
            self.base_url = "https://api.moonshot.cn/v1"
            self.model = "kimi-k2-0905-preview"
            self.max_tokens = 256 * 1024
        elif llm_vendor == "deepseek":
            self.base_url = "https://api.deepseek.com"
            self.model = "deepseek-chat"
            self.max_tokens = 128 * 1024
        elif llm_vendor == "openai":
            self.base_url = ""
            self.model = "gpt-5.2"
            self.max_tokens = 400000
        else:
            self.base_url = "http://jb-aionlineinferenceservice-149881696782669824-8000-default.incluster-prod.dros-new.zhejianglab.cn/v1"
            self.max_tokens = content_length * 1024
            self.model = "Qwen3-8B"

        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY") if llm_vendor != "vllm" else "",
            base_url=self.base_url,
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
            api_url = self.base_url.replace("/v1", "/tokenize")
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
