# Test with new usage case
FROM python:3.11-slim as compiler
ENV PYTHONUNBUFFERED 1

# Copy 
ADD . .
#
RUN python -m venv /venv 

# Enable venv
ENV PATH="/venv/bin:$PATH"

RUN pip install . && \ 
    pip install torch torchvision torchaudio fastapi vietocr pdf2image opencv-python uvicorn[standard] python-multipart && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# Runtime stage
FROM python:3.11-slim as runner
# Copy 
COPY ./deploy .
#
RUN apt-get update && apt-get install -y curl ffmpeg libsm6 libxext6 libgl1-mesa-glx libegl1-mesa libopengl0
#
COPY --from=compiler /venv /venv
#
ENV PATH="/venv/bin:$PATH"
#
CMD uvicorn api:app --port 9000 --host 0.0.0.0