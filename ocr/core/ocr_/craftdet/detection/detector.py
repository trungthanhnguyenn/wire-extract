
import time
import torch
import numpy as np
from ..models.utils import load_model
from ..models import CraftNet, RefineNet
from ..utils import image as image_utils
from ..detection import utils as craft_utils


class Detector:
  """
  
  """

  def __init__(
    self, 
    craft: str = '~/weights/craft/mlt25k.pth', 
    refiner: str = '~/weights/craft/refinerCTW1500.pth', 
    use_cuda: bool = True
  ):
    """
    
    """
    self.use_cuda = use_cuda
    #
    self.craft_net = load_model(CraftNet(), craft, use_cuda)
    self.craft_net.share_memory()
    #
    try:
      self.refine_net = load_model(RefineNet(), refiner, use_cuda)
      self.refine_net.share_memory()
    except:
      self.refine_net = None

  def detect(
    self, 
    image, 
    long_size: int = 1024, 
    low_text: float = 0.4, 
    text_thresh: float = 0.7, 
    link_thresh: float = 0.4,
    poly: bool = True
  ):
    """
    Detect text from inputs.
    :param image:       input to be detect.
    :param long_size:   longest image size for inference.
    :param low_text:    text low-bound score.
    :param text_thresh: text confidence threshold.
    :param link_thresh: link confidence threshold.
    :param poly:        enable polygon type.
    """
    t0 = time.time()
    # read/convert image
    image = image_utils.read_image(image)
    # resize
    img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(image, long_size)
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()
    # preprocessing
    x = image_utils.norm_mean_var(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = torch.autograd.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if self.use_cuda and torch.cuda.is_available():
        x = x.cuda()
    preprocessing_time = time.time() - t0
    t0 = time.time()
    # forward pass
    with torch.no_grad():
        y, feature = self.craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()
    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    # refine link
    if self.refine_net is not None:
        with torch.no_grad():
            y_refiner = self.refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    refinenet_time = time.time() - t0
    t0 = time.time()
    # Post-processing
    boxes, polys = craft_utils.get_det_boxes(score_text, score_link, text_thresh, link_thresh, low_text, poly)
    # coordinate adjustment
    boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)
    #polys = [x for x in polys if x is not None]
    polys = craft_utils.adjust_result_coordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]
    # get image size
    img_height = image.shape[0]
    img_width = image.shape[1]
    # calculate box coords as ratios to image size
    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])
    boxes_as_ratio = np.array(boxes_as_ratio)
    # calculate poly coords as ratios to image size
    polys_as_ratio = []
    for poly in polys:
        polys_as_ratio.append(poly / [img_width, img_height])
    #polys_as_ratio = np.array(polys_as_ratio)
    # generate heatmap
    text_score_heatmap = image_utils.cv2_heatmap_image(score_text)
    link_score_heatmap = image_utils.cv2_heatmap_image(score_link)
    # calculate time
    postprocess_time = time.time() - t0
    times = {
      "resize_time": resize_time,
      "preprocessing_time": preprocessing_time,
      "craftnet_time": craftnet_time,
      "refinenet_time": refinenet_time,
      "postprocess_time": postprocess_time,
    }
    # return results.
    return {
      "boxes": boxes,
      "boxes_as_ratios": boxes_as_ratio,
      "polys": polys,
      "polys_as_ratios": polys_as_ratio,
      "heatmaps": {
        "text_score_heatmap": text_score_heatmap,
        "link_score_heatmap": link_score_heatmap,
      },
      "times": times,
    }
