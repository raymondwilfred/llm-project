import json

with open('d:/LLM/Chapter 2/tools.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        has_wikipedia = any('import wikipedia' in line for line in cell['source'])
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if 'import os' in line and not has_wikipedia:
                new_source.append('import wikipedia\n')
                new_source.append("wikipedia.set_user_agent('MyLangchainAgent/1.0 (contact@example.com)')\n")
                has_wikipedia = True
        cell['source'] = new_source

with open('d:/LLM/Chapter 2/tools.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
