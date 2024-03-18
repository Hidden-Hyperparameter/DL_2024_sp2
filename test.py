import torch
import torchvision
# torch.backends.cudnn.enabled = False
import os
device = 'cuda' if torch.cuda.is_available() else 'XU WEI'
def test(classIndex,name='test',model_name = './models/cifar10_4x_best.pth'):
    model = torch.load(model_name)
    p = os.path.join('testdataset',name,f'class{classIndex}')
    _,_,filename=list(os.walk(p))[0]
    from PIL import Image
    import numpy as np
    NORMALIZES = ([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])
    tensors = []
    for file in filename:
        img=Image.open(os.path.join(p,file))
        img_rgb = img.convert('RGB')
        tensor  =torch.from_numpy(np.array(img_rgb).transpose((2,0,1))/255)
        for i in range(3):
            tensor[i]=(tensor[i]-NORMALIZES[0][i])/NORMALIZES[1][i]
        tensors.append(tensor.reshape(1,3,128,128))
    bsize = 100
    batches = []
    i = 0
    while True:
        try:
            final_tensor = torch.cat(tuple(tensors[i*bsize:i*bsize+bsize]),dim=0)
            batches.append(final_tensor)
        except RuntimeError:
            break
        i+=1
    assert len(batches)>=10, 'Dataset has some problems'
    assert batches[0].shape==torch.Size([100,3,128,128]), 'Invalid figures'
    accs = 0
    for batch in batches:
        _, predicted = torch.max(model(batch.to(device)), 1)
        leng = len(batch)
        acc = (predicted==(classIndex*torch.ones(leng)).to(device)).sum().item()/leng
        accs += acc
    accs /= 10
    return accs
def fulltest(model_name='./models/cifar10_4x_best.pth'):
    tot_acc = 0
    for index in range(10):
        tot_acc+= test(index,model_name=model_name)
    print('The accuracy of model is',tot_acc*10,'%')
    return tot_acc*10
if __name__ == '__main__':
    fulltest()