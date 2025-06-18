import os
import cv2
import argparse

from tqdm import tqdm
from PIL import Image
from pathlib import Path

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ocr_.craftdet.detection import Detector
from ocr_.utils import pdf2imgs, bbox2ibox, cv2crop, cv2drawbox


argparsers = argparse.ArgumentParser()
argparsers.add_argument('--file', default='~/Downloads/sample.pdf')
argparsers.add_argument('--out', default='~/Downloads/sample')


if __name__ == "__main__":
  #
  args = argparsers.parse_args()
  images = pdf2imgs(args.file)
  lsd = cv2.createLineSegmentDetector()
  #
  out_dir = Path(args.out).expanduser().resolve()
  os.makedirs(out_dir, exist_ok=True)
  #
  detector = Detector(
    craft=os.getcwd() + '/weights/craft/mlt25k.pth',
    refiner=os.getcwd() + '/weights/craft/refinerCTW1500.pth',
    use_cuda=True
  )
  #
  config = Cfg.load_config_from_name('vgg_transformer')
  config['weights'] = str(Path(os.getcwd() + '/weights/ocr/vgg_transformer.pth').expanduser().resolve())
  config['device'] = 'cpu'
  ocr = Predictor(config)
  #
  for i in range(len(images)):
    image = images[i]
    # Template image.
    # line_img = lineDetect(image)
    # line_img_path = os.path.join(out_dir, '{}_line.jpg'.format(i))
    #
    texts = []
    z = detector.detect(image)
    same_line = None
    for j in tqdm(range(len(z['boxes'])), desc='Process page ({}/{})'.format(i + 1, len(images))):
      ib = bbox2ibox(z['boxes'][j])
      img = cv2crop(image, ib[0], ib[1])

      text = ocr.predict(Image.fromarray(img))
      if ib[0] == same_line:
        texts[-1] += ' ' + text
      else:
        same_line = ib[0]
        texts.append(text)
      image = cv2drawbox(image, ib[0], ib[1])
    # Output image.
    img_path = os.path.join(out_dir, '{}.jpg'.format(i))
    img_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img_out)
    
    #line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(line_img_path, line_img)
    # Text logs.
    log_path = os.path.join(out_dir, '{}.txt'.format(i))
    with open(log_path, 'w') as f:
      for line in texts:
        f.write("%s\n" % line)
