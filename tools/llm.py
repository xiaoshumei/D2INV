import os

from openai import OpenAI
import tiktoken

import requests
import json


class LLM:
    def __init__(
        self,
        llm_vendor="vllm",
    ):
        assert llm_vendor in ["kimi", "deepseek", "openai","qwen", "vllm"]
        print(f"Using LLM vendor: {llm_vendor}")
        self.llm_vendor = llm_vendor
        if llm_vendor == "kimi":
            self.base_url = "http://i-nhi.zhejianglab.org/maas/v1"
            self.model = "Kimi-K3"
        elif llm_vendor == "deepseek":
            self.base_url = "http://i-nhi.zhejianglab.org/maas/v1"
            self.model = "DeepSeek-V4-Flash"
        elif llm_vendor == "qwen":
            self.base_url = "http://i-nhi.zhejianglab.org/maas/v1"
            self.model = "Qwen3.5-397B-A17B"
        elif llm_vendor == "openai":
            self.base_url = "http://43.159.131.233:3001/v1"
            self.model = "gpt-5.6-sol"
        else:
            self.base_url = "http://jb-aionlineinferenceservice-157166493733771136-8000-nhss-job.v5000-prod.nhss.zhejianglab.com/v1"
            self.model = "Qwen3.6-27B"

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
