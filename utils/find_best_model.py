import json
import os
base_dir = './models'
lst = list(os.walk(base_dir))[0][2]
best_test_acc = 0
best_file = None
for file in lst:
    do_write = True
    if not file.endswith('.json'):continue
    print(f'trying file {file}')
    path = os.path.join(base_dir,file)
    print(path)
    f = open(path,'r')
    try:
        dic = json.load(f)
    except json.decoder.JSONDecodeError:
        print(f'json file {file} is broken')
        do_write = False
        continue
    else:
        print(f'json file {file} is loaded successfully')
    test_acc = 0
    try:
        test_acc = dic['test_acc']
    except KeyError:
        continue
    if test_acc is not None and test_acc > best_test_acc:
        best_file = file
        best_test_acc = test_acc
print('-----------------------------')
print('best file is',best_file)
print('best test acc is ',best_test_acc)