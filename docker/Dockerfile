FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install python and utils
RUN apt-get update && apt-get install -y python3-pip curl unzip \
    libosmesa-dev libglew-dev patchelf libglfw3-dev

# Get git
RUN apt-get update && apt-get install -y git

# Download mujoco
RUN curl https://www.roboti.us/download/mjpro150_linux.zip --output /tmp/mujoco.zip && \
    mkdir -p /root/.mujoco && \
    unzip /tmp/mujoco.zip -d /root/.mujoco && \
    rm -f /tmp/mujoco.zip

# Retrieve mujoco key
RUN curl https://www.roboti.us/file/mjkey.txt --output /root/.mujoco/mjkey.txt

# Set library load path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

# Copy the mujoco license key
# COPY mjkey.txt /root/.mujoco/mjkey.txt

# Get requirements file
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN git clone https://github.com/Farama-Foundation/d4rl.git && \
    cd d4rl && \
    pip3 install -e .

# Download policy eval code.
#RUN pip3 install gsutil
#RUN gsutil -m cp -r gs://gresearch/deep-ope/d4rl /tmp
#RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git;
#RUN cd google-research && \
#    git sparse-checkout init --cone && \
#    git sparse-checkout set policy_eval
#RUN set -e && set -x
#RUN python3 -m pip install -r google-research/policy_eval/requirements.txt


# Install gym with mujoco
# RUN pip3 install gym[mujoco]
# RUN pip3 install -r /tmp/requirements.txt

# Test the python packages
RUN python3 -c "import gym; gym.make('Humanoid-v2'); print('done')"