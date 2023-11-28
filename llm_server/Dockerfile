FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /workdir
COPY ./requirements.txt requirements.txt
RUN pip install --upgrade --no-cache-dir -r requirements.txt
ENV HUGGINGFACE_MODEL="openchat/openchat_3.5"
# Comment if you want to store the model weights in the host's disk
COPY ./download.py /workdir/download.py
RUN python download.py
COPY ./main.py /workdir/main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
