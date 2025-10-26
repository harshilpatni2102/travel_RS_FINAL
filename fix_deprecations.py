#!/usr/bin/env python3
"""Script to fix deprecation warnings"""

# Fix app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('use_container_width=True', 'width="stretch"')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed app.py")

# Fix utils.py
with open('utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('use_container_width=True', 'width="stretch"')

with open('utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed utils.py")
print("\nAll deprecation warnings fixed!")
