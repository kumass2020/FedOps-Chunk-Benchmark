FROM python:3.9.16
WORKDIR /app
# RUN apk add --no-cache gcc musl-dev linux-headers g++
# RUN dnf install gcc-c++ python3-devel

RUN apt-get install -y ntpclient
RUN echo "NTP_CLIENT_SYSTEM_CONF['ntp_servers'] = ['time.google.com']" >> /etc/ntp.conf

COPY ./requirements_server.txt requirements.txt
# COPY ./requirements_tf.txt requirements.txt

RUN pip install -r requirements.txt

RUN wandb login fa67dec671c4384afbb282bd683f98d443c6b1d1

COPY ./advanced_server.py server.py
COPY ./utils.py utils.py

COPY ./flwr ./flwr
COPY ./data ./data
EXPOSE 8080
ENTRYPOINT sh -c 'ntpclient -s -D time.google.com && python3 /app/server.py'

