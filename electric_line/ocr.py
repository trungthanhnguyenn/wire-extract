import numpy as np
import re
import os
import cv2
import urllib.request
from PIL import Image
from pathlib import Path
from electric_line.extract_electric_line import Tool


def ensure_craft_weights():
    root_dir = Path(__file__).resolve().parents[1] 
    weights_dir = root_dir / "weights" / "craft"
    weights_dir.mkdir(parents=True, exist_ok=True)

    craft_path = weights_dir / "mlt25k.pth"
    refiner_path = weights_dir / "refinerCTW1500.pth"

    if not craft_path.exists():
        print("⚠️  Missing craft_mlt_25k.pth, starting to download...")
        urllib.request.urlretrieve(
            "https://github.com/clovaai/CRAFT-pytorch/releases/download/1.0/craft_mlt_25k.pth",
            craft_path
        )
        print("✅ craft_mlt_25k.pth downloaded successfully")

    if not refiner_path.exists():
        print("⚠️   Missing refinerCTW1500.pth, starting to download...")
        urllib.request.urlretrieve(
            "https://github.com/clovaai/CRAFT-pytorch/releases/download/1.0/refiner_CTW1500.pth",
            refiner_path
        )
        print("✅ refinerCTW1500.pth downloaded successfully")

    return str(craft_path), str(refiner_path)



class OCR:
    def __init__(self, output_dir, image_path, detector, ocr, tool: Tool):
        self.output_dir = output_dir
        self.image_path = image_path
        self.detector = detector
        self.ocr = ocr
        self.tool = tool

    def sliding_window_tiles(self, img, tile_size=512, overlap=128):
        h, w = img.shape[:2]
        stride = tile_size - overlap
        # ocr_txt_path = Path(self.output_dir) / (Path(self.image_path).stem + ".txt")
        # # ocr_txt_path = os.path.join(self.output_dir, os.path.basename)
        # if ocr_txt_path.exists():
        #     print(f"✅ Finish OCR: {ocr_txt_path}, skip.")
        #     return []
        tiles = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                tile = img[y:y + tile_size, x:x + tile_size]
                tile_h, tile_w = tile.shape[:2]

                # Padding if tile moved out of image
                if tile_h < tile_size or tile_w < tile_size:
                    padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                    padded[:tile_h, :tile_w] = tile
                    tile = padded

                tiles.append({
                    "tile": tile,
                    "offset": (x, y)
                })

        return tiles

    # def ocr_on_tiles(self, img, text_pattern=r"\d{2,3}A", text_thresh=0.5, link_thresh=0.5):
    #     if np.mean(img) > 250:
    #         print("⚠️  Skip white background.")
    #         return []
        
    #     tiles = self.sliding_window_tiles(img)
    #     results = []

    #     for tile_info in tiles:
    #         tile = tile_info["tile"]
    #         ox, oy = tile_info["offset"]

    #         detections = self.detector.detect(tile, text_thresh=text_thresh, link_thresh=link_thresh)
    #         # for (x1, y1), (x2, y2) in detections.get("boxes", []):
    #         for box in detections.get("boxes", []):
    #             box_np = np.array(box)
    #             x_coords = box_np[:, 0]
    #             y_coords = box_np[:, 1]
    #             x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
    #             x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))

    #             h, w = tile.shape[:2]
    #             x1, x2 = np.clip([x1, x2], 0, w)
    #             y1, y2 = np.clip([y1, y2], 0, h)

    #             crop = tile[y1:y2, x1:x2]
    #             if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
    #                 continue

    #             text = self.ocr.predict(Image.fromarray(crop))
    #             if re.fullmatch(text_pattern, text):
    #                 global_box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
    #                 results.append({
    #                     "text": text,
    #                     "box": global_box,
    #                     "center": (
    #                         (global_box[0] + global_box[2]) // 2,
    #                         (global_box[1] + global_box[3]) // 2
    #                     )
    #                 })

    #     return results
    
    # def _process_crop_batch(self, crops, crop_metas, text_pattern):
    #     batch_texts = self.ocr.predict_batch(crops)  # batching
    #     batch_results = []
    #     for text, (ox, oy, x1, y1, x2, y2) in zip(batch_texts, crop_metas):
    #         if re.fullmatch(text_pattern, text):
    #             global_box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
    #             batch_results.append({
    #                 "text": text,
    #                 "box": global_box,
    #                 "center": (
    #                     (global_box[0] + global_box[2]) // 2,
    #                     (global_box[1] + global_box[3]) // 2
    #                 )
    #             })
    #     return batch_results
    
    def _process_crop_batch(self, crops, crop_metas, text_pattern):
        batch_texts = self.ocr.predict_batch(crops)
        batch_results = []

        for text, (ox, oy, x1, y1, x2, y2) in zip(batch_texts, crop_metas):
            matches = re.findall(text_pattern, text)
            if matches:
                global_box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)

                for match in matches:
                    batch_results.append({
                        "text": match,
                        "box": global_box,
                        "center": (
                            (global_box[0] + global_box[2]) // 2,
                            (global_box[1] + global_box[3]) // 2
                        )
                    })

        return batch_results

    def ocr_on_tiles(self, img, text_pattern=r"\d{2,3}A", text_thresh=0.4, link_thresh=0.4, batch_size=8):
        ocr_txt_path = Path(self.output_dir) / (Path(self.image_path).stem + ".txt")
        if ocr_txt_path.exists():
            print(f"✅ Finished OCR: {ocr_txt_path}, skip.")
            return []
        
        if np.mean(img) > 250:
            print("⚠️  White background skip.")
            return []

        tiles = self.sliding_window_tiles(img)
        results = []

        crops = []
        crop_metas = []

        for tile_info in tiles:
            tile = tile_info["tile"]
            ox, oy = tile_info["offset"]

            detections = self.detector.detect(tile, text_thresh=text_thresh, link_thresh=link_thresh)
            for box in detections.get("boxes", []):
                box_np = np.array(box)
                x_coords = box_np[:, 0]
                y_coords = box_np[:, 1]
                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))

                h, w = tile.shape[:2]
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)

                crop = tile[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                crops.append(Image.fromarray(crop))
                crop_metas.append((ox, oy, x1, y1, x2, y2))

                if len(crops) == batch_size:
                    results.extend(self._process_crop_batch(crops, crop_metas, text_pattern))
                    crops, crop_metas = [], []

        # Final batch
        if crops:
            results.extend(self._process_crop_batch(crops, crop_metas, text_pattern))

        return results

    def deduplicate(self, results, iou_thresh=0.3):
        dedup = []
        for current in results:
            if all(self.tool.compute_iou(current['box'], other['box']) < iou_thresh for other in dedup):
                dedup.append(current)
        dedup.sort(key=lambda x: (x["center"][1], x["center"][0]))  # sort by y, then x
        return dedup
    

def draw_highlighted_boxes(image, results):
    for item in results:
        box = item["box"]
        x1, y1, x2, y2 = map(int, box)
        text = item["text"]
        # Green bbox + white backround
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image
    
# def save_ocr_results_with_visualization(image_path, output_dir, results):
#     """
#     Saves:
#     1. A .txt file listing detected text and coordinates.
#     2. An annotated image with bounding boxes and text.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(image_path))[0]

#     # === Save .txt file ===
#     txt_path = os.path.join(output_dir, f"{base_name}_ocr.txt")
#     with open(txt_path, 'w') as f:
#         for item in results:
#             text = item["text"]
#             x1, y1, x2, y2 = item["box"]
#             f.write(f"{text} ({x1}, {y1}), ({x2}, {y2})\n")

#     # === Save annotated image ===
#     image = cv2.imread(image_path)
#     for item in results:
#         text = item["text"]
#         x1, y1, x2, y2 = item["box"]
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    
#     img_output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
#     cv2.imwrite(img_output_path, image)

#     return txt_path, img_output_path

def save_ocr_results_with_visualization(image_path, output_dir, results):
    img = cv2.imread(image_path)
    output_img = draw_highlighted_boxes(img, results)

    out_path_img = os.path.join(output_dir, Path(image_path).stem + "_ocr_vis.jpg")
    out_path_txt = os.path.join(output_dir, Path(image_path).stem + ".txt")

    cv2.imwrite(out_path_img, output_img)
    with open(out_path_txt, 'w') as f:
        for i, r in enumerate(results):
            f.write(f"{i+1:02d}. {r['text']} at center {r['center']}, box={r['box']}\n")

    print(f"💾 Saved visual result to {out_path_img}")
    print(f"💾 Saved OCR text to {out_path_txt}")
