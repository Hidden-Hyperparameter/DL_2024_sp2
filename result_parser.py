# import os
# l = os.listdir('./models')
# max_acc = 0
# best_l = l
# lst = []
# for c in l:
#     if c.endswith('.pth') and ('.' in c[-9:-4]):
#         acc = c[-9:-4]
#         if(not acc[0].isdigit()):
#             acc = acc[1:]
#         acc = float(acc)
#         if acc>max_acc:
#             max_acc=acc
#             lst=[c]
#         elif abs(acc-max_acc)<1e-2:
#             max_acc=acc
#             lst.append(c)
# print('max accuracy',max_acc)
# print('corosponding files',lst)
# exit()

import os
file = 'cifar10_4x_best.json'
try_dirs = ['.','./bests']
dic = {}
for try_dir in try_dirs:
    full_dir = os.path.join(try_dir,file)
    f = None
    try:
        f=open(full_dir,'r')
    except:
        pass
    else:
        import json
        dic = json.load(f)
        break
history = dic['train_history']
optimizers = []
train_accs = []
valid_accs = []
for one_10_epochs in history:
    optimizers.append(one_10_epochs['optimizer'])
    train_accs.extend([c['train accuracy'] for c in one_10_epochs['results']])
    valid_accs.extend([c['valid accuracy'] for c in one_10_epochs['results']])
# print(len(optimizers))
# print(len(train_accs))
# exit()
import matplotlib.pyplot as plt
import numpy as np
leng = len(train_accs)
x = np.linspace(0,leng,leng)
# print(x.shape)
y = np.array(train_accs)
# print(y.shape)
z = np.array(valid_accs)
# exit()
plt.plot(x,y,label='train accuracy')
plt.plot(x,z,label='valid accuracy')
plt.legend()
plt.savefig('./result.png')