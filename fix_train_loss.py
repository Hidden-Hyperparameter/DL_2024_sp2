import json
import os
base_dir = './models'
lst = list(os.walk(base_dir))[0][2]
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
    history = dic['train_history']
    try:
        history[0]['results']
    except (TypeError,KeyError):
        do_write = False
        print(f'file {file} is invalidly formatted')
        pass
    else:
        for one_some_epochs in history:
            for c in one_some_epochs['results']:
                try:
                    c['loss']/=4
                except KeyError:
                    continue
    if do_write:
        ff = open(path,'w')
        json.dump(dic,ff,indent=4)
    print(f'finish file {file}')
    # break