import torch
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict


def init_weights(modules):
  """
  Initial weights of modules.
  :param modules:   modules to be initialized.
  """
  for m in modules:
    if isinstance(m, nn.Conv2d):
      torch.nn.init.xavier_uniform_(m.weight.data)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.01)
      m.bias.data.zero_()


def copy_state_dict(state_dict):
    """
    Copy state dict into new one that fit craft networks.
    :param state_dict:  input state dict.
    :return:            fitted state dict.
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_model(model, weight_path: str, cuda: bool = True):
  """
  Load weight into model.
  :param model:       target model.
  :param weight_path: path to weight file.
  :param cuda:        use cuda or not.
  :return:            model with new weight.
  """
  weight_path = Path(weight_path).expanduser().resolve()
  weight_path.parent.mkdir(exist_ok=True, parents=True)
  weight_path = str(weight_path)
  # load model into device.
  if torch.cuda.is_available() and cuda:
    model.load_state_dict(copy_state_dict(torch.load(weight_path)))
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = False
  else:
    model.load_state_dict(copy_state_dict(torch.load(weight_path, map_location='cpu')))
  model.eval()
  return model
