FROM python:3
USER root

RUN apt-get update
# RUN apt-get -y install locales && \
#     localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
# ENV LANG ja_JP.UTF-8
# ENV LANGUAGE ja_JP:ja
# ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# ENV API_KEY=${API_KEY}
# ENV API_KEY_SECRET=${API_KEY_SECRET}
# ENV BEARER_TOKEN=${BEARER_TOKEN}
# ENV ACCESS_TOKEN=${ACCESS_TOKEN}
# ENV ACCESS_TOKEN_SECRET=${ACCESS_TOKEN_SECRET}
COPY requirements.txt /tmp/

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN python -m pip install jupyterlab
RUN pip install -r /tmp/requirements.txt
# abcdefg
