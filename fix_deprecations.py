#!/usr/bin/env python3
"""Script to fix deprecation warnings"""

# Fix app.py - Replace use_container_width with width parameter
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Plotly chart - replace use_container_width=True with width='stretch'
content = content.replace('st.plotly_chart(fig, use_container_width=True)', "st.plotly_chart(fig, width='stretch')")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed app.py - Replaced use_container_width with width='stretch'")

# Fix utils.py if needed
with open('utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('use_container_width=True', "width='stretch'")

with open('utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed utils.py")
print("\nAll Streamlit deprecation warnings fixed!")
