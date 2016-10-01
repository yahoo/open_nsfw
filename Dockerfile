FROM kaixhin/caffe
RUN mkdir -p /opt/open_nsfw
ADD . /opt/open_nsfw/
WORKDIR /opt/open_nsfw
