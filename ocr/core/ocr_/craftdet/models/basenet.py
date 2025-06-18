import torch
from torchvision import models
from .utils import init_weights
from collections import namedtuple


class BaseNet(torch.nn.Module):
  """
  CRAFT base network is based on VGG16 backbone.
  """
  
  def __init__(
      self,
      freeze: bool = True,
      pretrained: bool = True, 
      weights: str = "VGG16_BN_Weights.IMAGENET1K_V1"):
    """
    Create base network.
    :param freeze:      freeze the first convolution layer.
    :param pretrained:  use pretrained weights or not.
    :param weights:     pretrained weights to be used.
    """
    super().__init__()
    # make vgg16 features.
    features = models.vgg16_bn(weights=weights).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    self.slice5 = torch.nn.Sequential()
    for x in range(12): # conv2_2
      self.slice1.add_module(str(x), features[x])
    for x in range(12, 19): # conv3_3
      self.slice2.add_module(str(x), features[x])
    for x in range(19, 29): # conv4_3
      self.slice3.add_module(str(x), features[x])
    for x in range(29, 39): # conv5_3
      self.slice4.add_module(str(x), features[x])
    # fc6, fc7 without atrous conv
    self.slice5 = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
      torch.nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
      torch.nn.Conv2d(1024, 1024, kernel_size=1)
    )
    # init weights.
    if not pretrained:
      init_weights(self.slice1.modules())
      init_weights(self.slice2.modules())
      init_weights(self.slice3.modules())
      init_weights(self.slice4.modules())
    # no pretrained model for fc6 and fc7.
    init_weights(self.slice5.modules())  
    if freeze:
      for param in self.slice1.parameters():  # only first conv
        param.requires_grad= False      

  def forward(self, x):
    """
    Forward function.
    :param x:   inputs of network.
    :return:    outputs of network.
    """
    h = self.slice1(x)
    h_relu2_2 = h
    h = self.slice2(h)
    h_relu3_2 = h
    h = self.slice3(h)
    h_relu4_3 = h
    h = self.slice4(h)
    h_relu5_3 = h
    h = self.slice5(h)
    h_fc7 = h
    vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
    return vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
