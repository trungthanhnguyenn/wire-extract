import cv2
import numpy as np


def read_image(image):
  """
  Read an image from file, bytes or numpy array.
  :param image: image source.
  :return:      numpy image.
  """
  img = image
  # read image from path.
  if type(image) == str:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # read image from bytes.
  elif type(image) == bytes:
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # read image from numpy array.
  elif type(image) == np.ndarray:
    if len(image.shape) == 2:  # grayscale
      img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
      img = image
    elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
      img = image[:, :, :3]
  # return image.
  return img


def norm_mean_var(rgb_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
  """
  Normalize RGB image by mean variance.
  :param rgb_img:   input RGB image.
  :param mean:      mean.
  :param variance:  variance.
  :return:          normalized image.
  """
  img = rgb_img.copy().astype(np.float32)
  img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
  img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
  return img


def denorm_mean_var(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
  """
  Denormalize RGB image from mean variance.
  :param in_img:    input normalized image.
  :param mean:      mean.
  :param variance:  variance.
  :return:          denormalized image.
  """
  img = in_img.copy()
  img *= variance
  img += mean
  img *= 255.0
  img = np.clip(img, 0, 255).astype(np.uint8)
  return img


def resize_aspect_ratio(img, long_size, interpolation = cv2.INTER_CUBIC):
  """
  Resize but keep aspect ratio.
  :param img:           input image.
  :param long_size:     desired size.
  :param interpolation: interpolation algorithm.
  :return:              resized image.
  """
  height, width, channel = img.shape
  # set target image size
  target_size = long_size
  ratio = target_size / max(height, width)
  target_h, target_w = int(height * ratio), int(width * ratio)
  proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
  # make canvas and paste image
  target_h32, target_w32 = target_h, target_w
  if target_h % 32 != 0:
    target_h32 = target_h + (32 - target_h % 32)
  if target_w % 32 != 0:
    target_w32 = target_w + (32 - target_w % 32)
  resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
  resized[0:target_h, 0:target_w, :] = proc
  target_h, target_w = target_h32, target_w32
  # calculate size heatmap.
  size_heatmap = (int(target_w / 2), int(target_h / 2))
  # return results.
  return resized, ratio, size_heatmap


def cv2_heatmap_image(img):
  """
  Convert to heatmap image.
  :param img: input image.
  :return:    headmap image.
  """
  img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
  img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
  return img
