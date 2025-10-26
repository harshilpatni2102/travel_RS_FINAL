#!/usr/bin/env python3
"""Script to fix Plotly chart warnings"""

import re

# Fix app.py - Replace width='stretch' with use_container_width=True
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all st.plotly_chart with width parameter to use_container_width
content = re.sub(
    r"st\.plotly_chart\(fig,\s*width='stretch'\)",
    "st.plotly_chart(fig, use_container_width=True)",
    content
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed all st.plotly_chart calls in app.py")
print("\nAll Plotly warnings should now be resolved!")
