import os

def replace_hex_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements
    # Violet/Purple tones
    content = content.replace('#7c3aed', '#e11d48')  # rose-600
    content = content.replace('#6d28d9', '#be123c')  # rose-700
    content = content.replace('#ede9fe', '#ffe4e6')  # rose-100
    content = content.replace('#0f0a1a', '#292524')  # stone-800
    
    # Cyan/Blue tones
    content = content.replace('#06b6d4', '#f59e0b')  # amber-500
    content = content.replace('#ecfeff', '#fef3c7')  # amber-100

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

def main():
    src_dir = os.path.join('d:\\', 'Code-Editors', 'VS-Code', 'Python', 'Skin_App', 'frontend', 'src')
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.tsx') or file.endswith('.ts') or file.endswith('.css'):
                filepath = os.path.join(root, file)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            
                # Replacements
                new_content = content.replace('#7c3aed', '#e11d48')
                new_content = new_content.replace('#6d28d9', '#be123c')
                new_content = new_content.replace('#ede9fe', '#ffe4e6')
                new_content = new_content.replace('#0f0a1a', '#292524')
                
                new_content = new_content.replace('#06b6d4', '#f59e0b')
                new_content = new_content.replace('#ecfeff', '#fef3c7')
            
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated: {filepath}")

if __name__ == "__main__":
    main()
