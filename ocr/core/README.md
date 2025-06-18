# Python OCR + Retification
Cross-platform implementation of CRAFT: Character Region Awareness For Text detection.
## Python usage
First, install package from for this repo.
Please notice that this code below run for GPU-supported version
```
pip install .
pip install torch torchvision torchaudio fastapi vietocr pdf2image opencv-python uvicorn[standard] python-multipart
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

## Run deploying
### With uvicorn
```
uvicorn deploy.api:app --port 9000 --host 0.0.0.0
```
#### Usage
First open proxy that logging into terminal, then it's oke if seeing the notification 
```
{"Hello": "Main Apps"}
```
Next, click to the address and add the following postfix: `/docs`, you will the the main page of FastAPI
Currently, there exist 5 main functions:
(1) post a image
(2) get the **original** image
(3) get **image** from the prediction of retification model
(4) get **image** from the prediction of OCR model
(5) get **text** from the prediction of OCR model

### With gradio
```
gradio deploy/deploy_gradio.py
```
#### Usage
Just opening the address from gradio, this will lead you to predefined UI, feel free to test

## Features
Implemented feature
- [x] Create UI using gradio
- [x] Create API using FastAPI
- [ ] Create public proxy when using FastAPI
- [ ] Get prediction from docker image to local when using API
- [ ] Convert all of model to ONNX format
 

## License
[Apache License 2.0](LICENSE).<br>
Copyright &copy; 2023 [Hieu Pham](https://github.com/hieupth). All rights reserved.