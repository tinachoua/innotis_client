import ast

txt_file = 'coco80.txt'

txt_file = 'imagenet1000.txt'

with open(txt_file, 'r') as f:
    cnt = f.read()
    label = ast.literal_eval(cnt)

print(type(label))
print(label.items()[:5])