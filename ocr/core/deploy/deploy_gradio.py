from pathlib import Path
import gradio as gr
import time
import cv2
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
import re

from PIL import Image
from pathlib import Path
from tqdm import tqdm


from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ocr_.craftdet.detection.detector import Detector
from ocr_.preprocessor.model import DewarpTextlineMaskGuide
from ocr_.utils import pdf2imgs, bbox2ibox, cv2crop, cv2drawbox, cv2drawboxtext

DEFAULT_SIZE_IMAGE = 512
save_origin_path = Path(os.getcwd() + "/prediction/origin").expanduser().resolve()

########
# Rerctification
########

#
# recti_model = DewarpTextlineMaskGuide(image_size=DEFAULT_SIZE_IMAGE)
# recti_model = torch.nn.DataParallel(recti_model)
# state_dict = torch.load(os.getcwd() + '/weights/rectification/30.pt', map_location='cuda:0')
#
# recti_model.load_state_dict(state_dict)
# recti_model.cuda()
save_rectif_path = Path(os.getcwd()+ "/prediction/rectification").expanduser().resolve()

########
# OCR
########

lsd = cv2.createLineSegmentDetector()
#
detector = Detector(
    craft=os.getcwd() + '/weights/craft/mlt25k.pth',
    refiner=os.getcwd() + '/weights/craft/refinerCTW1500.pth',
    use_cuda=True
)
#
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = str(Path(os.getcwd() + '/weights/ocr/vgg_transformer.pth').expanduser().resolve())
config['device'] = 'cuda:0'
ocr = Predictor(config)
save_ocr_path = Path(os.getcwd()+ "/prediction/ocr").expanduser().resolve()

##########

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def predict(img_intput, save_path, filename, recti_model):

    if not os.path.exists(save_path): 
        print('Create non-existed Save Path')
        os.makedirs(save_path)

    img_size = DEFAULT_SIZE_IMAGE

    img = np.array(img_intput)[:, :, :3] / 255.
    img_h, img_w, _ = img.shape
    input_img = cv2.resize(img, (img_size, img_size))

    with torch.no_grad():
        recti_model.eval()
        input_ = torch.from_numpy(input_img).permute(2, 0, 1).cuda()
        input_ = input_.unsqueeze(0)
        start = time.time()

        bm = recti_model(input_.float())
        bm = (2 * (bm / 223.) - 1) * 0.99
        ps_time = time.time() - start

    bm = bm.detach().cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()  # h * w * 2

    out = F.grid_sample(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
    img_geo = ((out[0] * 255.).permute(1, 2, 0).numpy()).astype(np.uint8)

    cv2.imwrite(filename, img_geo[:, :, ::-1])  # save

    return img_geo[:, :, ::-1], ps_time

def run(file_paths):
    ########
    # Process
    ########

    total_time = 0.0

    # verify file_paths:
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    # clear output folder
    os.system('rm -rf {}'.format(save_origin_path))
    os.makedirs(save_origin_path, exist_ok=True)
    os.system('rm -rf {}'.format(save_rectif_path))
    os.makedirs(save_rectif_path, exist_ok=True)
    os.system('rm -rf {}'.format(save_ocr_path))
    os.makedirs(save_ocr_path, exist_ok=True)

    # run
    start = time.time()
    for file_path in file_paths:
        namefile_only: str = file_path.split('/')[-1]
        name_file = namefile_only.rsplit('.', 1)[0]
        extension = namefile_only.rsplit('.', 1)[-1]

        images = []
        vid_writer = None
        # read file
        if extension == 'pdf':
            print("Converting PDF to images...")
            images = pdf2imgs(file_path)
        elif extension in ['jpg', 'jpeg', 'png']:
            print("Processing images...")
            images.append(Image.open(file_path))
        elif extension in ['mp4']:
            cap = cv2.VideoCapture(file_path)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Capturing video...")
            for i in range(nframes):
                ret, frame = cap.read()
                # save to folder and read again, images will contains path to images
                ori_img_path = os.path.join(save_origin_path, f'origin_image_{i}.jpg')
                cv2.imwrite(ori_img_path, np.asarray(frame))
                images.append(ori_img_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            save_path = Path(name_file).with_suffix('.mp4')  # force *.mp4 suffix on results videos
            vid_writer = cv2.VideoWriter(f"prediction/{save_path}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
        os.makedirs(os.path.join(save_origin_path, name_file), exist_ok=True)
        os.makedirs(os.path.join(save_rectif_path, name_file), exist_ok=True)
        os.makedirs(os.path.join(save_ocr_path, name_file), exist_ok=True)

        img_num = 0.0
        for idx, image in enumerate(images):  # img_names:  
            if extension in ['mp4']:
                # read image_path
                image = cv2.imread(image)

            # predict rectification
            # filename =  os.path.join(save_rectif_path, f"{name_file}_{idx}.jpg")
            # img_rectify, time_process = predict(image, save_rectif_path, filename, recti_model)
            image = np.ascontiguousarray(image)
            img_rectify = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # total_time += time_process
            img_num += 1

            # predict OCR
            texts = []
            z = detector.detect(img_rectify, text_thresh=0.5, link_thresh=0.5)
            same_line = None
            important_symbols = []
            # batch_img_rectify_crop = []
            for j in tqdm(range(len(z['boxes'])), desc='Process page ({}/{})'.format(idx + 1, len(images))):
                ib = bbox2ibox(z['boxes'][j])
                img_rectify_crop = cv2crop(img_rectify, ib[0], ib[1])
                # batch_img_rectify_crop.append(Image.fromarray(img_rectify_crop))
                #TODO: draw both box and text
                text = ocr.predict(Image.fromarray(img_rectify_crop))
                if re.match(r"\d{2,3}A", text):  # như 25A, 50A, 40A,...
                    center = (
                        (ib[0][0] + ib[1][0]) // 2,
                        (ib[0][1] + ib[1][1]) // 2,
                    )
                    important_symbols.append({
                        "text": text,
                        "center": center,
                        "box": ib,
                    })
                # texts.append(text)
                if same_line and abs(ib[0][1] - same_line[1]) < 10:  # 10 pixels threshold for same line
                    texts[-1] += ' ' + text
                else:
                    same_line = ib[0]
                    texts.append(text)
                # img_rectify = cv2drawbox(img_rectify, ib[0], ib[1])
                img_rectify = cv2drawboxtext(img_rectify, text, ib[0], ib[1])
                
            # texts = ocr.predict_batch(batch_img_rectify_crop)

            # Output image.
            if extension in ['jpg', 'jpeg', 'png', 'pdf']:
                ori_img_path = os.path.join(save_origin_path, name_file, f'{idx}.jpg')
                img_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(ori_img_path, np.asarray(img_out))
                img_path = os.path.join(save_ocr_path, name_file, f'{idx}.jpg')
                # img_out = cv2.cvtColor(img_rectify, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path, img_rectify)
            
                # Text logs.
                log_path = os.path.join(save_ocr_path, name_file, 'content_{}.txt'.format(idx))
                with open(log_path, 'w') as f:
                    for line in texts:
                        f.write("%s\n" % line)

            if vid_writer is not None:
                vid_writer.write(img_rectify)

            # print('FPS: %.1f' % (1.0 / (total_time / img_num)))
        if vid_writer is not None:
            log_path = os.path.join(save_ocr_path, 'video_content.txt'.format(idx))
            with open(log_path, 'w') as f:
                for i, line in enumerate(texts):
                    f.write(f"Frame {i}: {str(line)}\n")
            vid_writer.release()
            cv2.destroyAllWindows()
        entime = time.time() - start
        print("Total time for prediction: ", entime)
    
    return [
        glob.glob(str(save_origin_path) + '/*/*.jpg', recursive=True),
        glob.glob(str(save_ocr_path) + '/*/*.txt', recursive=True),
        glob.glob(str(save_rectif_path) + '/*/*.jpg', recursive=True),
        glob.glob(str(save_ocr_path) + '/*/*.jpg', recursive=True),
    ]


if __name__ == '__main__':

    with gr.Blocks() as demo:
        file_output  = gr.Files()
        with gr.Row():
            upload_button = gr.UploadButton("Upload a file or files", file_count="multiple")
            trans_button = gr.Button("Predict")

        with gr.Row():
            original = gr.Gallery(
                label="Original Image", show_label=True, columns=1, object_fit="contain", format="jpg"
            )
            content_detect = gr.Files(None, label="Content detection of OCR")

        with gr.Row():
            rectification_predict = gr.Gallery(
                label="Rectification Prediction", show_label=True, columns=1, object_fit="contain", format="jpg"
            )
            ocr_predict = gr.Gallery(
                label="OCR Prediction", show_label=True, columns=1, object_fit="contain", format="jpg"
            )
    
        upload_button.upload(upload_file, upload_button, file_output)
        
        trans_button.click(
            fn=run, 
            inputs=file_output, 
            outputs=[
                original, 
                content_detect,
                rectification_predict, 
                ocr_predict,               
            ]
        )
    print("Running server...")
    demo.launch(share=True, server_name="0.0.0.0")