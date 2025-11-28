import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import Linear
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from label import Waterlevel


# Testing the saved model
path = 'resnet-best.pt'

resnet_true = torchvision.models.resnet50(pretrained=False)        # Switch to the ResNet model and run it
resnet_true.classifier = Linear(2048, 10)
model = resnet_true

print(model)
if torch.cuda.device_count() > 1:
    print("Using Multiple GPUs")
    model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load(path))

# Read the data in the test set
root_test = r' D:\Gato Code\Test1’
test_dataset = Waterlevel(root_test)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
with torch.no_grad():
    for batch_idx, (img, target, paths) in enumerate(test_dataloader):
        img, target = img.to(device), target.to(device)
        test_output = model(img)
        test_output = test_output.to(torch.float32
        test_target = target.to(torch.float32)
        _, predicted = torch.max(test_output.data, dim=1)
        print(paths)
        print(test_output)
        print(predicted)
        print(test_target)
        break

