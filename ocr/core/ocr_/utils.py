from pathlib import Path
import pdf2image
import numpy as np
import cv2
import argparse


def bbox2ibox(points):
    min_x, min_y = min(points[:, 0]), min(points[:, 1])
    max_x, max_y = max(points[:, 0]), max(points[:, 1])
    return (int(min_x), int(min_y)), (int(max_x), int(max_y))


def pdf2imgs(path):
    path = Path(path).expanduser().resolve()
    imgs = pdf2image.convert_from_path(str(path))
    imgs = [np.array(x) for x in imgs] if isinstance(imgs, list) else [imgs]
    return imgs


def cv2crop(img, a, b):
    crop = img[a[1] : b[1], a[0] : b[0]]
    return crop


def cv2drawbox(img, a, b):
    img = cv2.rectangle(img, a, b, color=(255, 0, 0), thickness=2)
    return img

def cv2drawboxtext(img, text, a, b):
    # dramw box
    # img = cv2.rectangle(img, a, b, color=(255, 0, 0), thickness=2)

    # draw text
    from PIL import ImageFont, ImageDraw, Image
    font = ImageFont.truetype("font-times-new-roman/SVN-Times New Roman 2.ttf", 20)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # https://www.blog.pythonlibrary.org/2021/02/02/drawing-text-on-images-with-pillow-and-python/
    bbox = draw.textbbox(a, text, font=font, anchor='ls')

    draw.rectangle([a, b], fill="yellow", width=2) # draw bbox detection 
    draw.rectangle(bbox, fill="yellow") # draw text detection
    draw.text(a, text, font=font, anchor='ls', fill="black")

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


def lineDetect(img):
    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blank = img * 0
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imshow("edge", edges)
    # cv2.waitKey()
    # This returns an array of r and theta values
    _ = cv2.HoughLinesP(
        edges,  # Input edge image
        5,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=10,  # Max allowed gap between line for joining them
    )
    return blank


# Document Image Rectification
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
