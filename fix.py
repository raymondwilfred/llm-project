import json

with open('d:/LLM/Chapter 2/Chain.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if '# base_url =' in line:
                line = line.replace('# base_url = "https://api.groq.com/openai/v1", ', '')
            new_source.append(line)
        cell['source'] = new_source

with open('d:/LLM/Chapter 2/Chain.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
