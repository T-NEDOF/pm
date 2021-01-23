# Dockerfile - this is a comment. Delete me if you want.
FROM python:3.7
COPY . /
WORKDIR /
# RUN pip install --upgrade pip
# RUN apt-get install 'ffmpeg'\
#     'libsm6'\ 
#     'libxext6'  -y
RUN apt update
RUN apt install libgl1-mesa-glx -y

# RUN pip install /packages/neural_network_model/dist/neural_network_model-0.1.0.tar.gz
RUN pip install -r requirements1.txt
RUN pip list

ENTRYPOINT ["python3"]
CMD ["server.py", "run"]
