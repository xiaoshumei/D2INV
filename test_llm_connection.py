#!/usr/bin/env python
"""
LLM 连通性诊断脚本。

用法:
    python test_llm_connection.py            # 测试默认 deepseek
    python test_llm_connection.py kimi       # 测试 kimi
    python test_llm_connection.py openai     # 测试 openai
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

VENDORS = {
    "kimi": {
        "base_url": "http://i-nhi.zhejianglab.org/maas/v1",
        "model": "Kimi-K3",
    },
    "deepseek": {
        "base_url": "http://i-nhi.zhejianglab.org/maas/v1",
        "model": "DeepSeek-V4-Flash",
    },
    "openai": {
        "base_url": "http://43.159.131.233:3001/v1",
        "model": "gpt-5.6-sol",
    },
}

def test_vendor(vendor: str):
    cfg = VENDORS.get(vendor)
    if not cfg:
        print(f"❌ 未知 vendor: {vendor}")
        return False

    api_key = os.getenv("LLM_API_KEY", "")
    print(f"\n{'='*60}")
    print(f"测试 vendor: {vendor}")
    print(f"  base_url: {cfg['base_url']}")
    print(f"  model:    {cfg['model']}")
    print(f"  api_key:  {'已设置 (长度 %d)' % len(api_key) if api_key else '❌ 未设置'}")
    print(f"{'='*60}")

    if not api_key:
        print("❌ 请在 .env 中设置 LLM_API_KEY")
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

        # 1. 列出可用的模型（检查 endpoint 是否可达）
        print("\n[1] 检查 /models endpoint...")
        try:
            models = client.models.list()
            model_ids = [m.id for m in models.data[:10]]
            print(f"  ✅ 可访问，共 {len(models.data)} 个模型。前 10 个:")
            for mid in model_ids:
                print(f"    - {mid}")
        except Exception as e:
            print(f"  ⚠ /models 失败: {e}")

        # 2. 调用 chat completion 测试
        print("\n[2] 测试 chat.completions...")
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": "说一个字：好"}],
                max_tokens=10,
            )
            content = resp.choices[0].message.content
            print(f"  ✅ 成功! 响应: {content!r}")
            return True
        except Exception as e:
            print(f"  ❌ 调用失败: {e}")
            print("\n  排查建议:")
            print("   - 如果 404: 模型名可能错误。用上面的 /models 列表核对")
            print("   - 如果 401/403: API key 无效或过期")
            print("   - 如果连接超时: base_url 网络不可达")
            return False

    except ImportError:
        print("❌ 请先安装 openai: pip install openai")
        return False

if __name__ == "__main__":
    vendor = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    ok = test_vendor(vendor)
    sys.exit(0 if ok else 1)