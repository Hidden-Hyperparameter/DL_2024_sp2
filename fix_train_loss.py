import json
import os
base_dir = './models'
lst = list(os.walk(base_dir))[0][2]
for file in lst:
    if not file.endswith('.json'):continue
    print(f'trying file {file}')
    path = os.path.join(base_dir,file)
    print(path)
    f = open(path,'r')
    dic = json.load(f)
    history = dic['train_history']
    for one_some_epochs in history:
        for c in one_some_epochs['results']:
            c['loss']/=4
    # ff = open(path,'w')
    # json.dump(dic,ff,indent=4)
    print(f'finish file {file}')
    break