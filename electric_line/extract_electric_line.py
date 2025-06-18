from electric_line.seg_de import Segmentation
from electric_line.calculate_length import LengthCalculator
import os
import cv2
import numpy as np
import re
from skimage.morphology import skeletonize
from skimage.util import invert
from ocr.core.ocr_.craftdet.detection.detector import Detector
from PIL import Image

class Tool:
    def __init__(self, image_path, output_dir, seg: Segmentation, calc: LengthCalculator):
        self.image_path = image_path
        self.output_dir = output_dir
        self.seg = seg
        self.calc = calc

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

    def get_electric_system(self):
        self.result = os.path.join(self.output_dir, f'result_{os.path.basename(self.image_path)}')
        if os.path.exists(self.result):
            return self.result

        circle_path, _ = self.seg.detect_circle()
        circle_img = cv2.imread(circle_path)

        if circle_img is None:
            raise ValueError("❌ self.circle_overlay does not exist!")

        line_img = self.seg.segment_line()

        # Chỉ chồng những pixel khác trắng
        non_white_mask = np.any(circle_img < [250, 250, 250], axis=2)
        result = line_img.copy()
        result[non_white_mask] = circle_img[non_white_mask]

        cv2.imwrite(self.result, result)
        print(f"✅ Final result saved to: {self.result}")
        return self.result

    def apply_mask_with_boxes(self) -> str:
        # read mask and base image
        orig = cv2.imread(self.image_path)
        mask = self.seg.visual()  # mask (white = electric wire, black = background)
        if mask is None or orig is None:
            raise FileNotFoundError("❌ Either mask nor original image ERROR!")

        if mask.shape != orig.shape[:2]:
            mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

        # overlay mask into white bg
        # white_bg = np.ones_like(orig, dtype=np.uint8) * 255
        white_bg = np.ones_like(orig, dtype=np.uint8) * 220
        result = cv2.bitwise_and(orig, orig, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        white_part = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
        final = cv2.add(result, white_part)

        # extract bbox from txt
        _, txt_path = self.seg.detect_circle()
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

        # draw bbox
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(final, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # save output
        output_path = os.path.join(self.output_dir, f'electric_line_{os.path.basename(self.image_path)}')
        cv2.imwrite(output_path, final)
        print(f"✅ Output saved to: {output_path}")
        return output_path
    
    def filter_straight_lines(self, min_length=13):
        """
        Args:
            min_length: minimum thick length to be accept as a line
        
        This fuction:    
            Delete pixels which does not thick enoungh to be accept as a line (vertical and horizontal).
        """
        output_path = os.path.join(self.output_dir, f'filtered_lines_{os.path.basename(self.image_path)}')

        # ✅ return if already exist
        if os.path.exists(output_path):
            print(f"✅ Already exist file at: {output_path}, skip.")
            return output_path

        # preprocessing mask
        mask_bin = self.seg.visual()
        mask = (mask_bin > 127).astype(np.uint8) * 255
        h, w = mask.shape
        output_mask = np.zeros_like(mask)

        # Loop in horizontal direction
        for y in range(h):
            count = 0
            for x in range(w):
                if mask[y, x] > 0:
                    count += 1
                else:
                    if count >= min_length:
                        output_mask[y, x - count:x] = 255
                    count = 0
            if count >= min_length:
                output_mask[y, w - count:w] = 255

        # Loop in vertical direction
        for x in range(w):
            count = 0
            for y in range(h):
                if mask[y, x] > 0:
                    count += 1
                else:
                    if count >= min_length:
                        output_mask[y - count:y, x] = 255
                    count = 0
            if count >= min_length:
                output_mask[h - count:h, x] = 255

        cv2.imwrite(output_path, output_mask)
        print(f"✅ Saved at: {output_path}")
        return output_path
    
    # def extract_boxes_from_txt(self, radius=10):
    #     """
    #     Đọc các dòng chứa 'Center = (x, y)' từ file txt để lấy bounding boxes.
    #     """
    #     _, txt_path = self.seg.detect_circle()
    #     boxes = []
    #     with open(txt_path, 'r') as f:
    #         for line in f.readlines():
    #             match = re.search(r'Center = \((\d+), (\d+)\)', line)
    #             if match:
    #                 cx, cy = int(match.group(1)), int(match.group(2))
    #                 box = (cx - radius, cy - radius, cx + radius, cy + radius)
    #                 boxes.append(box)
    #     return boxes  

    def extract_boxes_from_txt(self, radius=10, iou_thresh=0.3):
        """
        Args:
            radius: size of bbox
            iou_thresh: if iou between 2 bbox higher than iou_thresh -> delete, else -> keep
            
        Elliminated bbox which have (IOU > threshold).
        """
        _, txt_path = self.seg.detect_circle()
        raw_boxes = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                match = re.search(r'Center = \((\d+), (\d+)\)', line)
                if match:
                    cx, cy = int(match.group(1)), int(match.group(2))
                    box = (cx - radius, cy - radius, cx + radius, cy + radius)
                    raw_boxes.append(box)

        # ✅ filltered base of iou threshold
        filtered = []
        for box in raw_boxes:
            if all(self.compute_iou(box, other) < iou_thresh for other in filtered):
                filtered.append(box)

        return filtered

    def mask_within_union_box(self, mask, boxes, padding=10):
        """
        Args:
            mask: mask of electric trace in image
            boxes: list of bbox correspond to that image
            
        This function crop the area base on the coordinates of bounding box
        """
        h, w = mask.shape
        x_min = max(min([x1 for (x1, y1, x2, y2) in boxes]) - padding, 0)
        x_max = min(max([x2 for (x1, y1, x2, y2) in boxes]) + padding, w)
        y_min = max(min([y1 for (x1, y1, x2, y2) in boxes]) - padding, 0)
        y_max = min(max([y2 for (x1, y1, x2, y2) in boxes]) + padding, h)

        cropped = np.zeros_like(mask)
        cropped[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
        return cropped
    
    def thin_mask_to_1px(self, mask):
        binary = (mask > 127).astype(np.bool_)
        skeleton = skeletonize(binary).astype(np.uint8) * 255
        return skeleton

    def compute_length_by_pixel(self, skeleton):
        return np.count_nonzero(skeleton == 255)

    # def compute_length_by_contours(self, skeleton):
    #     contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     total_length = 0.0
    #     for contour in contours:
    #         if len(contour) < 10:
    #             continue  # bỏ qua contour quá ngắn
    #         for i in range(1, len(contour)):
    #             pt1 = contour[i - 1][0]
    #             pt2 = contour[i][0]
    #             total_length += np.linalg.norm(pt1 - pt2)
    #     return total_length


    def check_single_pixel_width(self, mask):
        """
        Args:
            mask:
        
        This function:
        Return list of white pixels which have more than 2 neighbor pixel  → not bad pixel
        """
        h, w = mask.shape
        bad_pixels = []

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if mask[y, x] == 255:
                    # check neighbor pixel 
                    neighbors = mask[y - 1:y + 2, x - 1:x + 2]
                    count = np.count_nonzero(neighbors == 255) - 1  # minus itself
                    if count > 2:
                        bad_pixels.append((x, y))

        return bad_pixels
    
    def calculate_length(self, input_mask_path):
        # load thin mask
        mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError("❌ Can not read mask.")
        
        # load bboxs file and extract
        txt_path = os.path.join(self.output_dir, f"cordinates_{os.path.basename(self.image_path)}.txt")
        if not os.path.exists(txt_path):
            _, txt_path = self.seg.detect_circle()
        boxes = self.extract_boxes_from_txt()
        if not boxes:
            raise ValueError("❌ Does not found any bounding box")

        
        masked_region = self.mask_within_union_box(mask, boxes, padding=15)
        skeleton = self.thin_mask_to_1px(masked_region)
        # test = self.check_single_pixel_width(skeleton)
        length_pixels = self.compute_length_by_pixel(skeleton)
        self.visualize_length_result(
                skeleton=skeleton,
                boxes=boxes,
                length_pixels=length_pixels)
        
        print(f"✅ Electric length (pixel): {length_pixels} px")
        print(f"Bad pixel count: {len(self.check_single_pixel_width(skeleton))}") 
        
        output_skeleton_path = os.path.join(self.output_dir, f"skeleton_{os.path.basename(self.image_path)}")
        cv2.imwrite(output_skeleton_path, skeleton)
        print(f"🖼️ Skeleton saved to: {output_skeleton_path}")

        return output_skeleton_path
    
    def visualize_length_result(self, skeleton, boxes, length_pixels):
        """
        Args:
            skeleton: thin mask of electric trace (1px thick)
            boxes: list bbox using cv2 match pattern
            length_pixels: length of the electric trace in pixel
        This module return a visualize image of the electric trace after extract and calculate length 
        """
        # load base image
        base = cv2.imread(self.image_path)
        if base is None:
            raise FileNotFoundError(f"❌ Not found image at path: {self.image_path}!")

        # load thin mask
        # skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            raise FileNotFoundError("❌ Not found mask!")
        
        # visualize electric line 1 pixel thick -> 5 pixel thick
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel ~ 5x5
        skeleton_dilated = cv2.dilate(skeleton, kernel, iterations=1)


        # pixel skeleton (white → green)
        skeleton_rgb = cv2.cvtColor(skeleton_dilated, cv2.COLOR_GRAY2BGR)
        skeleton_rgb[np.where((skeleton_rgb == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

        # Overlay skeleton into base image
        faded = (base * 0.35).astype(np.uint8) 
        # overlay = cv2.addWeighted(base, 0.8, skeleton_rgb, 0.5, 0)
        overlay = cv2.addWeighted(faded, 1.0, skeleton_rgb, 0.8, 0)

        # draw bboxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # calculate length in pixel
        text = f"Length (pixel): {length_pixels:.2f} px"
        cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # save and return
        vis_path = os.path.join(self.output_dir, f"visual_result_{os.path.basename(self.image_path)}")
        cv2.imwrite(vis_path, overlay)
        print(f"🖼️ Saved output at path: {vis_path}")
        return vis_path
    
    def extract_ampere_labels(self, detector, ocr_model):
        """
        Args:
            detector: module of craftdet
            orc_model: vietocr model
            
        This function:
            Detect and OCR all text using regrex match '50A', '25A', '40A'...
            Return list dict
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError("❌ Cannot read original image!")

        z = detector.detect(img, text_thresh=0.5, link_thresh=0.5)

        ampere_labels = []
        for box in z['boxes']:
            (x1, y1), (x2, y2) = box
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text = ocr_model.predict(Image.fromarray(crop))
            if re.match(r"\d{2,3}A", text):  # match regrex '25A', '50A'
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                ampere_labels.append({
                    "text": text,
                    "center": center,
                    "box": (x1, y1, x2, y2)
                })

        print(f"✅ Found {len(ampere_labels)} with label (A)")
        return ampere_labels
    
    def assign_ampere_to_lines(self, skeleton, ampere_labels, max_dist=30):
        """
        Args:
            skeleton: mask image
            ampere_labels: extracted label in OCR phase
            max_dist: maximum distance checking to assign the label to which electric wire
            
        This function:
            Which each '\d{2,3}A' pattern, assign to nearest electric wire (based on skeleton mask)
        """
        assigned = []
        white_points = np.column_stack(np.where(skeleton == 255))

        for label in ampere_labels:
            c_x, c_y = label["center"]
            dists = np.linalg.norm(white_points - np.array([c_y, c_x]), axis=1)
            if len(dists) == 0:
                continue
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_dist:
                label['matched_pixel'] = tuple(white_points[min_idx][::-1])  # (x, y)
                assigned.append(label)

        print(f"✅ Assigned successfully {len(assigned)} symbols on to nearest line.")
        return assigned

