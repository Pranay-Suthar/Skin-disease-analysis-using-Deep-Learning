import os
import re

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace classes
    new_content = content.replace('blue-', 'amber-')
    new_content = new_content.replace('violet-', 'rose-')
    new_content = new_content.replace('purple-', 'rose-')
    new_content = new_content.replace('cyan-', 'orange-')

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

def main():
    src_dir = os.path.join('d:\\', 'Code-Editors', 'VS-Code', 'Python', 'Skin_App', 'frontend', 'src')
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.tsx') or file.endswith('.ts') or file.endswith('.css'):
                replace_in_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
