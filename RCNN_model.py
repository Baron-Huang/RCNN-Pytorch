import torch
from torch import nn
from torchvision.models import vgg11
from torchsummaryX import summary

class VGG_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(VGG_Net, self).__init__()
        self.features_layers = base_net.features
        self.Averp = nn.AvgPool2d((2, 2), stride=2)
        self.Flat = nn.Flatten()
        self.Top_layer = nn.Linear(512, class_num)

    def forward(self, x):
        y = self.features_layers(x)
        y = self.Averp(y)
        y = self.Flat(y)
        self.features_vector = y
        y = self.Top_layer(y)
        return y

class VGG_Extractor(nn.Module):
    def __init__(self, base_net=None):
        super(VGG_Extractor, self).__init__()
        self.features_layers = base_net.features
        self.Averp = nn.AvgPool2d((2, 2), stride=2)
        self.Flat = nn.Flatten()

    def forward(self, x):
        y = self.features_layers(x)
        y = self.Averp(y)
        y = self.Flat(y)
        return y


class VGG_Probability(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(VGG_Probability, self).__init__()
        self.features_layers = base_net.features
        self.Averp = nn.AvgPool2d((2, 2), stride=2)
        self.Flat = nn.Flatten()
        self.Top_layer = nn.Linear(512, class_num)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        y = self.features_layers(x)
        y = self.Averp(y)
        y = self.Flat(y)
        y = self.Top_layer(y)
        y = self.Softmax(y)
        return y

class Regression_Net(nn.Module):
    def __init__(self, regression_num=None):
        super(Regression_Net, self).__init__()
        self.regression_layer_1 = nn.Linear(512, 64)
        self.relu_layer_1 = nn.ReLU()
        self.regression_layer_2 = nn.Linear(64, regression_num)

    def forward(self, x):
        y = self.regression_layer_1(x)
        y = self.relu_layer_1(y)
        y = self.regression_layer_2(y)
        return y

if __name__ == '__main__':
    base_net = vgg11(pretrained=False)
    vgg_net = VGG_Extractor(class_num=2, base_net=base_net)
    print(summary(vgg_net, torch.randn((2, 3, 96, 96))))

    #reg_net = Regression_Net(regression_num=4)
    #print(summary(reg_net, torch.randn((2, 512))))
    #print(vgg_net)





