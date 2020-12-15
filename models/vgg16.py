import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from envs.config import Config
from torchvision import models


class VGG16(nn.Module):
    def __init__(self, num_classes, is_flow=False, input_size=224, pretrained=True):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, # 0
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64), # 1
            nn.ReLU(inplace=True), # 2
            nn.Conv2d(in_channels=64, # 3
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64), # 4
            nn.ReLU(inplace=True), # 5
            nn.MaxPool2d(kernel_size=2, stride=2), # 6
            nn.Conv2d(in_channels=64, # 7
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128), # 8
            nn.ReLU(inplace=True), # 9
            nn.Conv2d(in_channels=128, # 10
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128), # 11
            nn.ReLU(inplace=True), # 12
            nn.MaxPool2d(kernel_size=2, stride=2), # 13
            nn.Conv2d(in_channels=128, # 14
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256), # 15
            nn.ReLU(inplace=True), # 16
            nn.Conv2d(in_channels=256, # 17
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256), # 18
            nn.ReLU(inplace=True), # 19
            nn.Conv2d(in_channels=256, # 20
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256), # 21
            nn.ReLU(inplace=True), # 22
            nn.MaxPool2d(kernel_size=2, stride=2), # 23
            nn.Conv2d(in_channels=256, # 24
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512), # 25
            nn.ReLU(inplace=True), # 26
            nn.Conv2d(in_channels=512, # 27
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512), # 28
            nn.ReLU(inplace=True), # 29
            nn.Conv2d(in_channels=512, # 30
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512), # 31
            nn.ReLU(inplace=True), # 32
            nn.MaxPool2d(kernel_size=2, stride=2), # 33
            nn.Conv2d(in_channels=512, # 34
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512), # 35
            nn.ReLU(inplace=True), # 36
            nn.Conv2d(in_channels=512, # 37
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512), # 38
            nn.ReLU(inplace=True), # 39
            nn.Conv2d(in_channels=512, # 40
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512), # 41
            nn.ReLU(inplace=True), # 42
            nn.MaxPool2d(kernel_size=2, stride=2), # 43
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        if is_flow:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=7*7*512, out_features=4096), # 0
                nn.ReLU(inplace=True), # 1
                nn.Dropout(p=0.9), # 2
                nn.Linear(in_features=4096, out_features=4096), # 3
                nn.ReLU(inplace=True), # 4
                nn.Dropout(p=0.8), # 5
                nn.Linear(in_features=4096, out_features=1000), # 6
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=7*7*512, out_features=4096), # 0
                nn.ReLU(inplace=True), # 1
                nn.Dropout(p=0.9), # 2
                nn.Linear(in_features=4096, out_features=4096), # 3
                nn.ReLU(inplace=True), # 4
                nn.Dropout(p=0.9), # 5
                nn.Linear(in_features=4096, out_features=1000), # 6
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return torch.sigmoid(self.classifier(x))

    def get_model(self):
        return self.model

    def set_parameter_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

class Model(nn.Module):
    def __init__(self, name='vgg16_bn', num_classes=101, is_flow=False):
        super(Model, self).__init__()
        if name == 'vgg16_bn':
            self.model = VGG16(num_classes=num_classes, is_flow=is_flow)
            ptr_model = models.vgg16_bn(pretrained=True)
            #print(ptr_model)
            self.model.load_state_dict(ptr_model.state_dict())

            if is_flow:
                self.model.features[0] = nn.Conv2d(20, 64, kernel_size=3, padding=1)
                num_ftrs = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
                self.model.load_state_dict(torch.load(Config.get_pretrained_weight_path('ucf101flow'))['state_dict'])
            self.set_parameter_requires_grad(self.model)

            num_in_ftrs = self.model.classifier[0].in_features
            num_out_ftrs = self.model.classifier[0].out_features
            self.model.classifier[0] = nn.Linear(num_in_ftrs, num_out_ftrs)

            num_in_ftrs = self.model.classifier[3].in_features
            num_out_ftrs = self.model.classifier[3].out_features
            self.model.classifier[3] = nn.Linear(num_in_ftrs, num_out_ftrs)
            
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, 1)
            print(self.model)
            
            print("Params to learn:")
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)

    def get_model(self):
        return self.model

    def set_parameter_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    import torch
    model = Model().get_model()
    x = torch.rand(32, 20, 224, 224)
    y = model(x)
    print(y.size())  # [1, 101] for UCF101
