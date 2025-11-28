import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from label import Waterlevel
import torchvision
from torch.nn import Linear
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. prepare data
root = r' D:\Gato Code\Training’
train_dataset = Waterlevel(root, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

root_val = r' D:\Gato Code\Training’
val_dataset = Waterlevel(root_val, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

# 2. load model
resnet_true = torchvision.models.resnet50(pretrained=True)
resnet_true.classifier = Linear(2048, 10)
model = resnet_true

if torch.cuda.device_count() > 1:
    print("Training with multiple GPUs...")
    model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 3. prepare super parameters
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)    #Add L2 regularization weight decay

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
        # output = output.to(torch.float32)
        # target = target.to(torch.float32)
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
    # print('epoch %d, loss: %f' % (epoch, loss.item()))

    # val
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
    torch.save(model.state_dict(), "resnet-last.pt")
    if acc_val == max(val_acc_list):
        torch.save(model.state_dict(), "resnet-best.pt")
        print("save epoch {} model".format(epoch))
    print("epoch = {},loss = {},acc = {},val_loss = {},acc_val = {}".format(epoch, train_loss, acc_train, val_loss_all, acc_val))
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score = {}".format(accuracy, precision, recall, f1))

print(epoch_list)
print(train_acc_list)
print(train_loss_list)
print(val_loss_list)
print(val_acc_list)
print("precision：", np.mean(precision_list))
print("recall：", np.mean(recall_list))
print("f1：", np.mean(f1_list))

# epoch
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
