FROM pytorch/pytorch

RUN apt-get update && apt install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt