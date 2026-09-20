"""
HTML Render Validator — Phase 4.

Provides lightweight HTML sanity checks for generated INV/infographic pages:
  - Detect missing ECharts CDN script
  - Detect empty chart divs with no associated script
  - Check for basic structural elements (head, body)
  - Detect common JS errors (missing closing tags, etc.)

These checks run AFTER the INV is assembled to flag obvious rendering problems
before showing the user a broken page.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def quick_html_sanity_check(html: str) -> Dict[str, Any]:
    """
    Perform a fast, non-browser-based sanity check on an HTML string.

    Returns a dict:
        {
            "valid": bool,
            "issues": [str, ...],
            "warnings": [str, ...],
        }
    """
    issues: List[str] = []
    warnings: List[str] = []

    if not html or not html.strip():
        return {"valid": False, "issues": ["Empty HTML"], "warnings": []}

    html_normalised = html.lower()

    # 1. Basic structural checks
    if "<html" not in html_normalised:
        warnings.append("Missing <html> tag (may still render in browsers)")
    if "</html>" not in html_normalised:
        warnings.append("Missing </html> closing tag")

    if "<head>" not in html_normalised and "<head " not in html_normalised:
        warnings.append("Missing <head> section")
    if "<body>" not in html_normalised and "<body " not in html_normalised:
        warnings.append("Missing <body> section")

    # 2. ECharts script CDN
    if "echarts" not in html_normalised:
        issues.append("No ECharts library loaded (missing <script> for echarts)")

    # 3. Chart div vs script
    chart_divs = re.findall(r'id=["\'](chart_\d+)["\']', html_normalised)
    chart_divs = list(set(chart_divs))

    for cd in chart_divs:
        div_id_present = f'id="{cd}"' in html_normalised or f"id='{cd}'" in html_normalised
        # Check if any script references this chart id
        referenced_in_script = cd in html_normalised
        if not referenced_in_script:
            # Try harder: look for plot_XXX pattern
            if not re.search(rf'plot_\d+\(.*?{re.escape(cd)}', html_normalised):
                warnings.append(f"Chart div #{cd} has no associated initialisation script")

    # 4. JS syntax check (basic)
    open_scripts = len(re.findall(r"<script[^>]*>", html_normalised))
    close_scripts = len(re.findall(r"</script>", html_normalised))
    if open_scripts != close_scripts:
        issues.append(f"Mismatched <script> tags: {open_scripts} open, {close_scripts} close")

    # 5. Check for common data injection bugs
    if "window.data" not in html_normalised and "window[" not in html_normalised:
        warnings.append("No window.data injection detected (charts may have no data source)")

    # 6. Check for empty rendering
    if "<div" in html_normalised and "echarts" not in html_normalised:
        if not re.search(r'<canvas|<svg', html_normalised):
            warnings.append("No canvas/svg elements detected; charts may not render")

    valid = len(issues) == 0
    return {"valid": valid, "issues": issues, "warnings": warnings}