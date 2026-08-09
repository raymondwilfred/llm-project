import json

with open('d:/LLM/Chapter 2/tools.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            line = line.replace('load_dotenv(override=True)', "load_dotenv(dotenv_path='../.env', override=True)")
            line = line.replace('load_dotenv()', "load_dotenv(dotenv_path='../.env', override=True)")
            new_source.append(line)
        cell['source'] = new_source

with open('d:/LLM/Chapter 2/tools.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
