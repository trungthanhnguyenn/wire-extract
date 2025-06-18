import cv2
import numpy as np
import os
import io
import torch
import torch.nn.functional as F
import pathlib

from PIL import Image
from tqdm import tqdm
import pdf2image
from fastapi import *
from fastapi.staticfiles import StaticFiles
import threading
import uvicorn

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ocr_.craftdet.detection import Detector
from ocr_.preprocessor.model import DewarpTextlineMaskGuide
from ocr_.utils import bbox2ibox, cv2crop, cv2drawbox


DEVICE = os.getenv('DEVICE', default="cuda:0")
IMAGE_SIZE = os.getenv('IMAGE_SIZE', default=224)

##################################
'''Function for prediction'''
def rectify_predict(img_intput, recti_model):

    img = np.array(img_intput)[:, :, :3] / 255.
    img_h, img_w, _ = img.shape
    input_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) 

    with torch.no_grad():
        recti_model.eval()
        input_ = torch.from_numpy(input_img).permute(2, 0, 1).cuda()
        input_ = input_.unsqueeze(0)

        bm = recti_model(input_.float())
        bm = (2 * (bm / 223.) - 1) * 0.99

    bm = bm.detach().cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()  # h * w * 2

    out = F.grid_sample(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
    img_geo = ((out[0] * 255.).permute(1, 2, 0).numpy()).astype(np.uint8)

    return img_geo[:, :, ::-1]

def ocr_predict(img_rectify, detector, ocr_model, idx, number_images):
    """idx, number_images: for tqdm progress bar"""
    texts = []
    z = detector.detect(img_rectify)

    batch_img_rectify_crop = []
    for j in tqdm(range(len(z['boxes'])), desc='Process page ({}/{})'.format(idx + 1, number_images)):
        ib = bbox2ibox(z['boxes'][j])
        img_rectify_crop = cv2crop(img_rectify, ib[0], ib[1])
        batch_img_rectify_crop.append(Image.fromarray(img_rectify_crop))
        img_rectify = cv2drawbox(img_rectify, ib[0], ib[1])

    texts = ocr_model.predict_batch(batch_img_rectify_crop)
    return img_rectify, texts

##################################
'''Load checkpoint and weight'''

# Rerctification
recti_model = DewarpTextlineMaskGuide(image_size=IMAGE_SIZE)
recti_model = torch.nn.DataParallel(recti_model)
state_dict = torch.load(os.getcwd() + '/weights/rectification/30.pt', map_location=DEVICE)
#
recti_model.load_state_dict(state_dict)
recti_model = recti_model.cuda() if "cuda" in DEVICE else recti_model.cpu()

# OCR
lsd = cv2.createLineSegmentDetector()
detector = Detector(
    craft=os.getcwd() + '/weights/craft/mlt25k.pth',
    refiner=os.getcwd() + '/weights/craft/refinerCTW1500.pth',
    use_cuda=True if "cuda" in DEVICE else False
)
#
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = str(os.getcwd() + '/weights/ocr/vgg_transformer.pth')
config['device'] = DEVICE
ocr_model = Predictor(config)

##################################
"""Define images app to store image after process"""
os.makedirs("./prediction", exist_ok=True)

ORIGIN_IMAGE_PATH = os.getenv('ORIGIN_IMAGE_PATH', default='origin_images')
ORIGIN_IMAGE_PATH = pathlib.Path("prediction/" + ORIGIN_IMAGE_PATH)
ORIGIN_IMAGE_PATH.mkdir(exist_ok=True)

RECTIFY_IMAGE_PATH = os.getenv('RECTIFY_IMAGE_PATH', default='refity_images')
RECTIFY_IMAGE_PATH = pathlib.Path("prediction/" + RECTIFY_IMAGE_PATH)
RECTIFY_IMAGE_PATH.mkdir(exist_ok=True)

OCR_IMAGE_PATH = os.getenv('OCR_IMAGE_PATH', default='ocr_images')
OCR_IMAGE_PATH = pathlib.Path("prediction/" + OCR_IMAGE_PATH)
OCR_IMAGE_PATH.mkdir(exist_ok=True)

OCR_TEXT_PATH = os.getenv('OCR_TEXT_PATH', default='ocr_text')
OCR_TEXT_PATH = pathlib.Path("prediction/" + OCR_TEXT_PATH)
OCR_TEXT_PATH.mkdir(exist_ok=True)

#
image_api = FastAPI()

# 
@image_api.get("/")
def read_app():
    return {"Hello": "Image Apps"}

image_api.mount('/ori_imgs', StaticFiles(directory=str(ORIGIN_IMAGE_PATH)), name='origin_images')
image_api.mount('/rectify_imgs', StaticFiles(directory=str(RECTIFY_IMAGE_PATH)), name='retification_images')
image_api.mount('/ocr_imgs', StaticFiles(directory=str(OCR_IMAGE_PATH)), name='ocr_images')
image_api.mount('/ocr_texts', StaticFiles(directory=str(OCR_TEXT_PATH)), name='oct_text')

##################################
'''Main App'''

#
app = FastAPI()
app.mount("/imageapi", image_api)

#
@app.get("/")
def root():
    return {"Hello": "Main Apps"}
#
@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    request_object_content = await file.read()
    extension = file.filename.split(".")[-1]

    images = []
    try: 
        images = []
        if extension in ["jpg", "jpeg", "png"]:
            images = [Image.open(io.BytesIO(request_object_content))]
        elif extension in ["pdf"]:
            images = pdf2image.convert_from_bytes(request_object_content)
    except:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")
    
    for i, img in enumerate(images):
        # predict rectification
        img_rectify = rectify_predict(img, recti_model)
        img_rectify = np.ascontiguousarray(img_rectify)
        
        # predict OCR
        img_ocr, texts = ocr_predict(img_rectify, detector, ocr_model, i, len(images)) 

        # save to image api
        cv2.imwrite(f"{ORIGIN_IMAGE_PATH}/{file.filename}_{i}.jpg", np.asarray(img))
        cv2.imwrite(f"{RECTIFY_IMAGE_PATH}/{file.filename}_{i}.jpg", img_rectify)
        cv2.imwrite(f"{OCR_IMAGE_PATH}/{file.filename}_{i}.jpg", img_ocr)
        with open(f"{OCR_TEXT_PATH}/{file.filename}_content{i}.txt", 'w') as f:
            for line in texts:
                f.write("%s\n" % line)

@app.get("/retify")
def get_retify():
    rectify_imgs = [f'/imageapi/rectify_imgs/{k}' for k in os.listdir(RECTIFY_IMAGE_PATH)]
    return {"retify": rectify_imgs}

@app.get("/ocr")
def get_ocr():
    ocr_imgs = [f'/imageapi/ocr_imgs/{k}' for k in os.listdir(OCR_IMAGE_PATH)]
    return {"ocr": ocr_imgs}

@app.get("/texts")
def get_texts():
    ocr_texts = [f'/imageapi/ocr_texts/{k}' for k in os.listdir(OCR_TEXT_PATH)]
    return {"texts": ocr_texts}

@app.get("/origin")
def get_origin():
    origin_imgs = [f'/imageapi/ori_imgs/{k}' for k in os.listdir(ORIGIN_IMAGE_PATH)]
    return {"retify": origin_imgs}

##################################