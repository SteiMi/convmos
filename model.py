from collections import OrderedDict
import torch
from torch import nn
from torch import Tensor
from typing import List, Tuple

from config_loader import config


class ConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        prev_layer_size: int,
        layer_size: int,
        linear: bool = False,
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            prev_layer_size,
            layer_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.ReLU()
        self.linear = linear

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)

        if not self.linear:
            out = self.act(out)

        return out


class GlobalNet(nn.Module):
    def __init__(self) -> None:
        super(GlobalNet, self).__init__()

        self.kernel_sizes = [
            int(k) for k in config.get('NN', 'kernel_sizes').split(",")
        ]
        self.layer_sizes = [int(k) for k in config.get('NN', 'layer_sizes').split(",")]
        # This uses all aux variables, the temperature/precipitation (+1), and elevation (+1)
        self.input_depth = (
            len(
                list(
                    filter(None, config.get('DataOptions', 'aux_variables').split(','))
                )
            )
            + 2
        )

        modules: List[Tuple[str, nn.Module]] = []
        prev_layer_size = self.input_depth
        for i, k in enumerate(self.kernel_sizes):

            # Always use ReLU except for the final layer
            is_layer_linear = (i == len(self.kernel_sizes) - 1)
            modules.append(
                (
                    'ConvBlock' + str(i),
                    ConvBlock(
                        k, prev_layer_size, self.layer_sizes[i], linear=is_layer_linear
                    ),
                )
            )
            prev_layer_size = self.layer_sizes[i]

        self.net: nn.Sequential = nn.Sequential(OrderedDict(modules))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        out = self.net(x)

        return out


class PerPixelLinear(nn.Module):

    def __init__(
        self, width: int, height: int, in_channels: int, bias: bool = True
    ) -> None:
        super(PerPixelLinear, self).__init__()
        self.width = width
        self.height = height
        self.in_channels = in_channels

        self.layer = nn.Conv1d(
            width * height,
            width * height,
            self.in_channels,
            groups=width * height,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, self.width * self.height, self.in_channels)

        x = self.layer(x)

        x = x.reshape(-1, self.width, self.height, 1)
        x = x.permute(0, 3, 1, 2)
        return x


class LocalNet(nn.Module):
    def __init__(self):
        super(LocalNet, self).__init__()

        # This uses all aux variables, the temperature/precipitation, and explicitely NOT elevation
        self.input_depth = (
            len(
                list(
                    filter(None, config.get('DataOptions', 'aux_variables').split(','))
                )
            )
            + 1
        )
        # We do not need elevation here, since it would always be the same for all times at each location
        self.input_width = config.getint('NN', 'input_width')
        self.input_height = config.getint('NN', 'input_height')

        # This should basically be an individual linear regression for each pixel/location
        self.local_net = PerPixelLinear(
            self.input_width, self.input_height, self.input_depth
        )

    def forward(self, x):
        # Insert everything except elevation
        return self.local_net(x[:, :-1])


char_to_module = {'l': LocalNet, 'g': GlobalNet}


class ConvMOS(nn.Module):
    def __init__(self, architecture: str = 'lgl'):
        super(ConvMOS, self).__init__()

        self.module_list = nn.ModuleList(
            [char_to_module[m_str]() for m_str in architecture]
        )

        self.out_act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:

        identity = x
        original_features = identity[:, 1:]
        new_input = identity[:, :1]

        for layer in self.module_list:

            resids = layer(x)

            # Basically reuse all the other features...
            original_input = x[:, :1]
            # ... and change temperature/precipitation according to residuals
            new_input = original_input + resids
            x = torch.cat([new_input, original_features], 1)

        return self.out_act(new_input)


if __name__ == "__main__":
    data = torch.rand(
        16,
        len(list(filter(None, config.get('DataOptions', 'aux_variables').split(','))))
        + 2,
        config.getint('NN', 'input_width'),
        config.getint('NN', 'input_height'),
    )

    model = ConvMOS()

    out = model(data)

    print(out.shape)
    print(out)
