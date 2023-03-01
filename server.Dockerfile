FROM python:3.9.16
WORKDIR /app
# RUN apk add --no-cache gcc musl-dev linux-headers g++
# RUN dnf install gcc-c++ python3-devel

COPY ./requirements_server.txt requirements.txt
# COPY ./requirements_tf.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./small_net_server.py server.py
#COPY ./tf_small_net_server.py server.py

COPY ./flwr ./flwr
EXPOSE 8080
ENTRYPOINT [ "python3", "/app/server.py" ]

