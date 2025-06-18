import sys
sys.path.append("/path/to/your/wire_extract/ocr/core") # -> TODO: replace the /path/to/your/ by your path

import os
from electric_line.extract_electric_line import Tool
from electric_line.calculate_length import LengthCalculator
from electric_line.seg_de import Segmentation
import cv2
import numpy as np
import re
from skimage.morphology import skeletonize
import glob

# TODO: Replace these paths with your actual paths
electric_dir = "/path/to/your/data/electric"
output_root = "/path/to/your/data/output"
label_dir = "/path/to/your/data/label"


def main():
    for image_path in glob.glob(os.path.join(electric_dir, "*.jpg")):
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # sample_1
        output_dir = os.path.join(output_root, image_name)

        os.makedirs(output_dir, exist_ok=True)

        print(f"🔄 Processing: {image_name}")

        # Initialize module
        seg = Segmentation(image_path, output_dir, label_dir)
        cal = LengthCalculator(seg)
        tool = Tool(image_path, output_dir, seg, cal)

        try:
            # mask_path = tool.get_electric_system()
            mask_path = tool.filter_straight_lines()
            mask = cv2.imread(mask_path)
            text = tool.calculate_length(mask_path)
        except Exception as e:
            print(f"❌ Cannot calculate length for {image_name}: {e}")
            continue

        print(f"✅ Finish: {image_name}")

if __name__=="__main__":
    main()