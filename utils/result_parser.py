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
name = 'cifar10_4x_03190942_acc92.39'
file = name + '.json'
try_dirs = ['.','./models']
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
valid_loses = []
train_loses = []
epochs = [0]
for one_some_epochs in history:
    optimizers.append(one_some_epochs['optimizer'])
    train_accs.extend([c['train accuracy'] for c in one_some_epochs['results']])
    valid_accs.extend([c['valid accuracy'] for c in one_some_epochs['results']])
    train_loses.extend([c['loss'] for c in one_some_epochs['results']])
    valid_loses.extend([c['valid loss'] for c in one_some_epochs['results']])
    epochs.append(epochs[-1]+one_some_epochs['epochs'])
# print(len(optimizers))
# print(len(train_accs))
plot = True
if not plot:
    hps = []
    for i,opt in enumerate(optimizers):
        lr,wd = opt['lr'],opt['weight_decay']
        if len(hps)==0 or not (abs(hps[-1][1]-lr)+abs(hps[-1][2]-wd)<5e-6):
            # try:
            #     print(hps[-1][1],lr,hps[-1][1],wd)
            #     print(abs(hps[-1][1]-lr)+abs(hps[-1][1]-wd))
            # except:
            #     pass
            hps.append((epochs[i],lr,wd))
    with open(f'hyperparameters_{name}.txt','w') as f:
        for u,v,w in hps:
            f.write('epoch {}: lr {:.6f}, weight_decay {:.6f}\n'.format(u,v,w))
else:
    import matplotlib.pyplot as plt
    import numpy as np
    leng = len(train_accs)
    # print(x.shape)
    y = np.array(train_accs)
    # print(y.shape)
    z = np.array(valid_accs)
    x = np.linspace(0,len(train_accs),len   (train_accs))
    # exit()
    plt.plot(x,y,label='train accuracy')
    plt.plot(x,z,label='valid accuracy')
    plt.legend()
    plt.savefig(f'./result_{name}.png')
    plt.clf()
    y = np.array(train_loses)
    # print(y.shape)
    z = np.array(valid_loses)
    plt.plot(x,y,label='training loss')
    plt.plot(x,z,label='validation loss')
    plt.legend()
    plt.savefig(f'./result_{name}_loss.png')
    print('done!')