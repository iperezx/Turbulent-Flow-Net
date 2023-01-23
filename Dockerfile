ARG cuda_version=11.2.2
ARG cudnn_version=8
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3-pip &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /turbulent-flow-net

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY TF-net TF-net
COPY Baselines Baselines
COPY Evaluation Evaluation
COPY data_gen.py data_gen.py

# CMD [ "python", "TF-net/run_model.py" ]
ENTRYPOINT ["tail", "-f", "/dev/null"]