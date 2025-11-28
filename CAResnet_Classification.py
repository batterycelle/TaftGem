import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from label import Waterlevel
import torchvision
from torch.nn import Linear
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

# 1. prepare data
root = r' D:\Gato Code\Training’
train_dataset = Waterlevel(root, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

root_val = r' D:\Gato Code\Training’
val_dataset = Waterlevel(root_val, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

# 2. load model    #ResNet50withCA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class ResNet50WithCoordAtt(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithCoordAtt, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # Remove the original fully connected layer
        self.resnet.fc = nn.Identity()
        # Add SE blocks after each residual block
        self.ca_blocks = nn.ModuleList([
            CoordAtt(256, 256),
            CoordAtt(512, 512),
            CoordAtt(1024, 1024),
            CoordAtt(2048, 2048)
        ])
        # Add a new fully connected layer
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.ca_blocks[0](x)

        x = self.resnet.layer2(x)
        x = self.ca_blocks[1](x)

        x = self.resnet.layer3(x)
        x = self.ca_blocks[2](x)

        x = self.resnet.layer4(x)
        x = self.ca_blocks[3](x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = ResNet50WithCoordAtt(num_classes=10)

# resnet_true = torchvision.models.resnet50(pretrained=True
# resnet_true.classifier = Linear(2048, 10)
# model = resnet_true
print(model)
if torch.cuda.device_count() > 1:
    print("Using Multiple GPUs...")
    model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
epoch_list = []
precision_list = []
recall_list = []
f1_list = []
for epoch in range(100):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    for batch_idx, (data, target, path) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(output.data, dim=1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

    acc_train = train_correct / train_total
    train_acc_list.append(acc_train)
    train_loss_list.append(train_loss)

    model.eval()
    correct = 0
    total = 0
    val_loss_all = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (data, target, path) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            val_loss = criterion(output, target.long())
            val_loss_all += val_loss.item()

            _, predicted = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    acc_val = correct / total

    val_acc_list.append(acc_val)
    val_loss_list.append(val_loss_all)
    epoch_list.append(epoch)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    precision_list.append(precision)
    recall = recall_score(y_true, y_pred, average='macro')
    recall_list.append(recall)
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_list.append(f1)


    # save model
    torch.save(model.state_dict(), "resnet50-CA-labelsmooth-last.pt")
    if acc_val == max(val_acc_list):
        torch.save(model.state_dict(), "resnet50-CA-labelsmooth-best.pt")
        print("save epoch {} model".format(epoch))
    print("epoch = {},loss = {},acc = {},val_loss = {},acc_val = {}".format(epoch, train_loss, acc_train, val_loss_all, acc_val))
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score = {}".format(accuracy, precision, recall, f1))

print(epoch_list)
print(train_acc_list)
print(train_loss_list)
print(val_loss_list)
print(val_acc_list)
print("accuracy：", np.mean(val_acc_list))
print("precision：", np.mean(precision_list))
print("recall：", np.mean(recall_list))
print("f1：", np.mean(f1_list))

#Draw the change graph for each epoch
fig, ax = plt.subplots()
line1 = ax.plot(epoch_list, train_loss_list, color='green', label="train_loss")
line3 = ax.plot(epoch_list, val_loss_list, color='blue', label='val_loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
plt.legend()
ax.spines['right'].set_visible(False)

z_ax = ax.twinx()
line2 = z_ax.plot(epoch_list, val_acc_list, color='red', label="val_acc")
line4 = z_ax.plot(epoch_list, train_acc_list, color='black', label="train_acc")
z_ax.set_ylabel('acc')

lns = line1+line2+line3+line4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.show()
