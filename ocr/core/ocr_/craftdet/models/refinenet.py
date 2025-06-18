import torch
import torch.nn as nn
from .utils import init_weights


class RefineNet(nn.Module):
  """
  Implementation of refine network.
  """

  def __init__(self):
    """
    CLass constructor.
    """
    super().__init__()
    # last convolution layer.
    self.last_conv = nn.Sequential(
      nn.Conv2d(34, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
    )
    # refine aspp layer 1.
    self.aspp1 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # refine aspp layer 2.
    self.aspp2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # refine aspp layer 3.
    self.aspp3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # refine aspp layer 4.
    self.aspp4 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # init weights.
    init_weights(self.last_conv.modules())
    init_weights(self.aspp1.modules())
    init_weights(self.aspp2.modules())
    init_weights(self.aspp3.modules())
    init_weights(self.aspp4.modules())

  def forward(self, y, upconv4):
    """
    Forward function.
    :param x:   inputs of network.
    :return:    outputs of network.
    """
    refine = torch.cat([y.permute(0,3,1,2), upconv4], dim=1)
    refine = self.last_conv(refine)
    aspp1 = self.aspp1(refine)
    aspp2 = self.aspp2(refine)
    aspp3 = self.aspp3(refine)
    aspp4 = self.aspp4(refine)
    #out = torch.add([aspp1, aspp2, aspp3, aspp4], dim=1)
    out = aspp1 + aspp2 + aspp3 + aspp4
    return out.permute(0, 2, 3, 1)  # , refine.permute(0,2,3,1)
