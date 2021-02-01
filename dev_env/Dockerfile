FROM pytorch/pytorch

RUN apt-get update && apt install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6

COPY requirements.txt /package/requirements.txt

RUN pip install -r --no-cache /packate/requirements.txt