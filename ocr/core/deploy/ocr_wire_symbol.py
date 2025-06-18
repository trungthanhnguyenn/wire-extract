import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np

from core.ocr_.craftdet.detection.detector import Detector
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from electric_line.ocr import save_ocr_results_with_visualization
from electric_line.ocr import OCR
from electric_line.extract_electric_line import Tool
from electric_line.calculate_length import LengthCalculator
from electric_line.seg_de import Segmentation

# === Config paths === -> TODO:
image_path = "/path/to/your/data/visual_result"
output_dir = "/path/to/your/data/output"
label_dir = "/path/to/your/data/label"

def main():
    assert os.path.exists(image_path), f"Image does not exist: {image_path}"

    # === Init models ===
    detector = Detector(
        craft=os.getcwd() + '/ocr/core/weights/craft/mlt25k.pth',
        refiner=os.getcwd() + '/ocr/core/weights/craft/refinerCTW1500.pth',
        use_cuda=True
    )

    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = str(os.getcwd() + '/ocr/core/weights/ocr/vgg_transformer.pth')
    config['device'] = 'cuda'
    ocr_model = Predictor(config)

    # === Tool setup ===
    seg = Segmentation(image_path, output_dir, label_dir)
    calc = LengthCalculator(seg)
    tool = Tool(image_path, output_dir, seg, calc)

    # === Run OCR detection ===
    image = cv2.imread(image_path)
    ocr_instance = OCR(output_dir, image_path, detector, ocr_model, tool)

    # raw_symbols = ocr_instance.ocr_on_tiles(image, detector, ocr_model)
    raw_symbols = ocr_instance.ocr_on_tiles(image, batch_size=24)
    deduped_symbols = ocr_instance.deduplicate(raw_symbols)

    # # === Show results ===
    # print(f"\n📌 Detected {len(deduped_symbols)} important symbols (after deduplication):")
    # for i, symbol in enumerate(deduped_symbols):
    #     text = symbol["text"]
    #     center = symbol["center"]
    #     box = symbol["box"]
        # print(f"{i+1:02d}. {text} at center {center}, box={box}")
        
    save_ocr_results_with_visualization(
        image_path=image_path,
        output_dir=output_dir,
        results=deduped_symbols)

if __name__ == "__main__":
    main()
