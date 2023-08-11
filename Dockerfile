FROM asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest
# FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-12.py310
WORKDIR /

COPY src /src

RUN pip install -r requirements.txt

ENTRYPOINT ["python3","src/train/train_script.py"]