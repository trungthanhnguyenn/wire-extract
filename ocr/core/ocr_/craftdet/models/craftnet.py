import torch
import torch.nn as nn
import torch.nn.functional as F
from .basenet import BaseNet
from .utils import init_weights


class DoubleConv(nn.Module):
  """
  Double of Conv, Batchnorm, ReLU blocks.
  """

  def __init__(self, in_ch, mid_ch, out_ch) -> None:
    """
    Class constructor.
    :param in_ch:   input channels.
    :param mid_ch:  middle channels.
    :param out_ch:  output channels.
    """
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
      nn.BatchNorm2d(mid_ch),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    """
    Forward function.
    :param x:   inputs of network.
    :return:    outputs of network.
    """
    return self.conv(x)
  

class CraftNet(nn.Module):
  """
  Implementation of CRAFT network.
  """
  
  def __init__(self, pretrained=False, freeze=False):
    super().__init__()
    # base network.
    self.basenet = BaseNet(pretrained=pretrained, freeze=freeze)
    # U-network.
    self.upconv1 = DoubleConv(1024, 512, 256)
    self.upconv2 = DoubleConv(512, 256, 128)
    self.upconv3 = DoubleConv(256, 128, 64)
    self.upconv4 = DoubleConv(128, 64, 32)
    # final cls.
    num_class = 2
    self.conv_cls = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
      nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
      nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
      nn.Conv2d(16, num_class, kernel_size=1),
    )
    # init weights.
    init_weights(self.upconv1.modules())
    init_weights(self.upconv2.modules())
    init_weights(self.upconv3.modules())
    init_weights(self.upconv4.modules())
    init_weights(self.conv_cls.modules())
        
  def forward(self, x):
    """
    Forward function.
    :param x:   inputs of network.
    :return:    outputs of network.
    """
    sources = self.basenet(x)
    # U-network.
    y = torch.cat([sources[0], sources[1]], dim=1)
    y = self.upconv1(y)
    # layer 2.
    y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
    y = torch.cat([y, sources[2]], dim=1)
    y = self.upconv2(y)
    # layer 3.
    y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
    y = torch.cat([y, sources[3]], dim=1)
    y = self.upconv3(y)
    # layer 4.
    y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
    y = torch.cat([y, sources[4]], dim=1)
    feature = self.upconv4(y)
    # final cls.
    y = self.conv_cls(feature)
    # return permute results.
    return y.permute(0,2,3,1), feature
