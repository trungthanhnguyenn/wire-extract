import os
import cv2
import numpy as np
from typing import List, Any
import re

class Segmentation:
    def __init__(self, image_path: str, output_dir: str, label_folder: str, angle_list: List[int] = [0, 90, 180, 270]):
        self.image_path = image_path
        self.output_dir = output_dir
        self.label_folder = label_folder
        self.angle_list = angle_list

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load image
        self.orig = cv2.imread(self.image_path)
        if self.orig is None:
            raise FileNotFoundError(f"❌ Image not found at {self.image_path}")
        self.gray = cv2.cvtColor(self.orig, cv2.COLOR_BGR2GRAY)

        # Inverted binary for processing
        _, self.bw_inv = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Placeholder for results
        self.line_overlay = None
        self.circle_overlay = None

    def segment_line(self, tol: float = 10.0, val: int = 42) -> Any:
        output_path = os.path.join(self.output_dir, f'line_{os.path.basename(self.image_path)}')
        if os.path.exists(output_path):
            return output_path
        dist = cv2.distanceTransform(self.bw_inv, cv2.DIST_L2, 5)
        norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        low = max(val - tol, 0)
        high = val + tol
        mask = ((norm >= low) & (norm <= high)).astype(np.uint8) * 255

        overlay = self.orig.copy()
        overlay[mask == 255] = (0, 0, 255)
        self.line_overlay = overlay

        # cv2.imwrite(output_path, overlay)

        coverage = np.sum(mask == 255) / mask.size * 100
        print(f"🟥 Line coverage: {coverage:.2f}%")
        # print(f"📁 Saved line overlay to: {output_path}")
        return overlay

    def rotate_image(self, image, angle: int):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    def detect_circle(self, threshold: float = 0.65, iou_thresh: float = 0.3) -> Any:
        def compute_iou_data(box1, box2):
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2

            xi1 = max(x1, x3)
            yi1 = max(y1, y3)
            xi2 = min(x2, x4)
            yi2 = min(y2, y4)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)
            union_area = area1 + area2 - inter_area

            return inter_area, area1, area2, inter_area / union_area if union_area > 0 else 0

        output_path = os.path.join(self.output_dir, f'circles_{os.path.basename(self.image_path)}')
        txt_path = os.path.join(self.output_dir, f'cordinates_{os.path.basename(self.image_path)}.txt')

        if os.path.exists(output_path) and os.path.exists(txt_path):
            return output_path, txt_path

        mask_template = np.zeros(self.gray.shape, dtype=np.uint8)
        overlay = self.orig.copy()
        # total_boxes = 0
        center_points = []
        existing_boxes = []

        for filename in os.listdir(self.label_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            template_path = os.path.join(self.label_folder, filename)
            template = cv2.imread(template_path)
            if template is None:
                print(f"⚠️ Skipping unreadable template: {template_path}")
                continue

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            for angle in self.angle_list:
                rotated = self.rotate_image(template_gray, angle)
                result = cv2.matchTemplate(self.gray, rotated, cv2.TM_CCOEFF_NORMED)
                loc = np.where(result >= threshold)

                h, w = rotated.shape
                for pt in zip(*loc[::-1]):
                    x1, y1, x2, y2 = pt[0], pt[1], pt[0] + w, pt[1] + h
                    new_box = (x1, y1, x2, y2)
                    new_area = (x2 - x1) * (y2 - y1)

                    keep = True
                    removed = []

                    for exist_box in existing_boxes:
                        inter_area, area1, area2, iou = compute_iou_data(new_box, exist_box)

                        if iou > iou_thresh and (
                            inter_area / area1 >= 0.8 or inter_area / area2 >= 0.8
                        ):
                            if area1 > area2:
                                removed.append(exist_box)
                            else:
                                keep = False
                                break

                    if keep:
                        for b in removed:
                            existing_boxes.remove(b)
                        existing_boxes.append(new_box)
                        cv2.rectangle(mask_template, (x1, y1), (x2, y2), 255, -1)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        center_points.append((center_x, center_y))
                        # total_boxes += 1

        with open(txt_path, 'w') as f:
            for idx, (cx, cy) in enumerate(center_points):
                f.write(f"Circle {idx + 1}: Center = ({cx}, {cy})\n")

        self.circle_overlay = overlay
        cv2.imwrite(output_path, overlay)

        # print(f"🟩 Detected {total_boxes} unique template matches.")
        print(f"📁 Saved circle overlay to: {output_path}")
        print(f"📄 Circle coordinates saved to: {txt_path}")
        return output_path, txt_path


    def detect_and_segment(self):
        self.result = os.path.join(self.output_dir, f'result_{os.path.basename(self.image_path)}')
        if os.path.exists(self.result):
            return self.result

        # circle_path = self.detect_circle()[0]
        circle_path, _ = self.detect_circle()
        circle_img = cv2.imread(circle_path)
        
        if circle_img is None:
            raise ValueError("❌ self.circle_overlay does not exist!")

        line_img = self.segment_line()

        green_mask = (
            (circle_img[:, :, 1] > 200) &
            (circle_img[:, :, 0] < 100) &
            (circle_img[:, :, 2] < 100) 
        )

        result = line_img.copy()
        result[green_mask] = circle_img[green_mask]

        cv2.imwrite(self.result, result)
        print(f"✅ Final result saved to: {self.result}")
        return self.result

    def visual(self):
        
        output_path = os.path.join(self.output_dir, f'masks_{os.path.basename(self.image_path)}')
        if os.path.exists(output_path):
            print(f"✅ Mask exist, skip: {output_path}")
            return cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

        segmented_img = self.segment_line()
        
        # debug_input_path = os.path.join(self.output_dir, f'debug_input_{os.path.basename(self.image_path)}')
        # cv2.imwrite(debug_input_path, segmented_img)
        
        hsv = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2HSV)
        
        l1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))      # lower
        l2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))   # upper
        mask_red = cv2.bitwise_or(l1, l2)
        
        # debug_red_path = os.path.join(self.output_dir, f'debug_red_raw_{os.path.basename(self.image_path)}')
        # cv2.imwrite(debug_red_path, mask_red)
        
        num, labels = cv2.connectedComponents(mask_red)
        clean_mask = np.zeros_like(mask_red)
        for i in range(1, num):
            component_size = (labels == i).sum()
            if component_size > 30:
                clean_mask[labels == i] = 255
        
        # debug_clean_path = os.path.join(self.output_dir, f'debug_clean_{os.path.basename(self.image_path)}')
        # cv2.imwrite(debug_clean_path, clean_mask)
        
        orig_gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if orig_gray is None:
            print(f"Image does not exist at: {self.image_path}")
            return
        
        if orig_gray.shape != segmented_img.shape[:2]:
            orig_gray = cv2.resize(orig_gray, (segmented_img.shape[1], segmented_img.shape[0]))
        
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        orig_cl = clahe.apply(orig_gray)
        _, bin_orig = cv2.threshold(orig_cl, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        dist = cv2.distanceTransform(bin_orig, cv2.DIST_L2, 5)
        d_th = 2.0
        mask_orig_thick = (dist > d_th).astype(np.uint8) * 255
        
        # debug_thick_path = os.path.join(self.output_dir, f'debug_thick_{os.path.basename(self.image_path)}')
        # cv2.imwrite(debug_thick_path, mask_orig_thick)
        
        # Morphological reconstruction
        seed = cv2.bitwise_and(clean_mask, mask_orig_thick)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # debug_seed_path = os.path.join(self.output_dir, f'debug_seed_{os.path.basename(self.image_path)}')
        # cv2.imwrite(debug_seed_path, seed)
        
        prev = np.zeros_like(seed)
        max_iter = 50 
        iteration = 0
        
        while iteration < max_iter:
            dil = cv2.dilate(seed, kernel)
            new_seed = cv2.bitwise_and(dil, mask_orig_thick)
            if np.array_equal(new_seed, seed):
                break
            prev = seed.copy()
            seed = new_seed
            iteration += 1
        
        print(f"Reconstruction stopped after {iteration} iterations")
        
        recon = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel, iterations=2)
        final = cv2.bitwise_or(recon, clean_mask)
        
        final = cv2.dilate(final, kernel, iterations=1)
        
        
        
        output_path = os.path.join(self.output_dir, f'masks_{os.path.basename(self.image_path)}')
        cv2.imwrite(output_path, final)
        
        print(f"Mask saved to: {output_path}")
        # print(f"Final mask has {np.sum(final > 0)} white pixels")
        
        return final
    

    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def apply_mask_with_boxes(self, mask_path) -> str:

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        orig = cv2.imread(self.image_path)

        if mask is None or orig is None:
            raise FileNotFoundError("❌ Either base image nor mask does not exist!")

        if mask.shape != orig.shape[:2]:
            mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

        white_bg = np.ones_like(orig, dtype=np.uint8) * 220
        result = cv2.bitwise_and(orig, orig, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        white_part = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
        final = cv2.add(result, white_part)

        _, txt_path = self.detect_circle()
        boxes = []

        with open(txt_path, 'r') as f:
            for line in f.readlines():
                match = re.search(r'Center = \((\d+), (\d+)\)', line)
                if match:
                    cx, cy = int(match.group(1)), int(match.group(2))
                    r = 10
                    box = (cx - r, cy - r, cx + r, cy + r)

                    # Check overlap
                    keep = True
                    for existing in boxes:
                        if self.compute_iou(box, existing) > 0.3:
                            keep = False
                            break

                    if keep:
                        boxes.append(box)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(final, (x1, y1), (x2, y2), (0, 0, 255), 2)

        output_path = os.path.join(self.output_dir, f'electric_line_{os.path.basename(self.image_path)}')
        cv2.imwrite(output_path, final)
        print(f"✅ Output saved to: {output_path}")
        return output_path



# class Tool:
#     def __init__(self, image_path: str, output_dir: str, label_folder: str):
#         pass
