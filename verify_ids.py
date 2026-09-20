#!/usr/bin/env python
"""
Cross-check that every getElementById target in web/app.js
exists as an id= in web/index.html.
"""
import re
from pathlib import Path

base = Path(__file__).resolve().parent / "web"
js = (base / "app.js").read_text(encoding="utf-8")
html = (base / "index.html").read_text(encoding="utf-8")

# All ids referenced in JS via getElementById
js_ids = set(re.findall(r'getElementById\("([^"]+)"\)', js))
html_ids = set(re.findall(r'id="([^"]+)"', html))

print("JS references:", sorted(js_ids := js_ids))
print("HTML defines :", sorted(html_ids))
print()

missing = sorted(js_ids - html_ids)
unused = sorted(html_ids - js_ids)

if not missing:
    print("✅ All JS getElementById references exist in HTML.")
else:
    print(f"❌ Missing in HTML: {missing}")

print("HTML ids not referenced in JS:", unused if unused else "none")