FROM python:3.9.16
WORKDIR /app
# RUN apk add --no-cache gcc musl-dev linux-headers g++
# RUN dnf install gcc-c++ python3-devel

COPY ./requirements.txt requirements.txt
#COPY ./requirements_tf.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./advanced_client.py client.py
COPY ./utils.py utils.py
#COPY ./tf_small_net_client.py client.py

COPY ./flwr ./flwr
COPY ./data ./data
EXPOSE 8080
# ENTRYPOINT [ "python3", "/app/client.py", "--partition", "$CLIENT_NUMBER" ]
ENTRYPOINT sh -c 'python3 /app/client.py --partition "$CLIENT_NUMBER"'
