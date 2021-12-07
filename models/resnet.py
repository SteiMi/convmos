from torch import nn, Tensor
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from config_loader import config


class ResNetBase(nn.Module):
    def __init__(self):
        super(ResNetBase, self).__init__()

        self.out_width = config.getint('NN', 'input_width')
        self.out_height = config.getint('NN', 'input_height')
        self.input_channels = len(list(filter(None, config.get('DataOptions', 'aux_variables').split(',')))) + 2

        self.base_model = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:

        x1 = self.base_model(x)
        return x1.reshape(-1, 1, self.out_width, self.out_height)


class ResNet18(ResNetBase):
    def __init__(self):
        super().__init__()

        self.base_model = resnet18()
        self.base_model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(512, self.out_width * self.out_height)


class ResNet34(ResNetBase):
    def __init__(self):
        super().__init__()

        self.base_model = resnet34()
        self.base_model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(512, self.out_width * self.out_height)


class ResNet50(ResNetBase):
    def __init__(self):
        super().__init__()

        self.base_model = resnet50()
        self.base_model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(2048, self.out_width * self.out_height)


class ResNet101(ResNetBase):
    def __init__(self):
        super().__init__()

        self.base_model = resnet101()
        self.base_model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(2048, self.out_width * self.out_height)


class ResNet152(ResNetBase):
    def __init__(self):
        super().__init__()

        self.base_model = resnet152()
        self.base_model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(2048, self.out_width * self.out_height)