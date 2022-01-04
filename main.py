import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('current cuda device is', device)

batch_size = 50
epoch_num = 15
learning_rate = 0.0001
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

print('number of training data: ', len(train_data))
print('number of test data: ', len(test_data))




image, label=train_data[0]

plt.imshow(image.squeeze().numpy(),cmap='gray')
plt.title('label : %s' % label)
plt.show()


train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

first_batch=train_loader.__iter__().__next__()
print('{:15s} | {:25s} | {}'.format('name','type','size'))
print('{:15s} | {:25s} | {}'.format('number of batch','',len(train_loader)))
print('{:15s} | {:25s} | {}'.format('first batch',str(type(first_batch)),len(first_batch)))
print('{:15s} | {:25s} | {}'.format('first batch[0]',str(type(first_batch[0])),first_batch[0].shape))
print('{:15s} | {:25s} | {}'.format('first batch[1]',str(type(first_batch[1])),first_batch[1].shape))
