import os

print('\t ...creating directories')
os.makedirs('./data/', exist_ok=True)
os.makedirs('./data/input', exist_ok=True)
os.makedirs('./data/output', exist_ok=True)
print('\t Done!')