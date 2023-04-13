FROM ubuntu:18.04

# Install python and utils
RUN apt-get update && apt-get install -y python3-pip curl unzip \
    libosmesa-dev libglew-dev patchelf libglfw3-dev

# Download mujoco
RUN curl https://www.roboti.us/download/mjpro150_linux.zip --output /tmp/mujoco.zip && \
    mkdir -p /root/.mujoco && \
    unzip /tmp/mujoco.zip -d /root/.mujoco && \
    rm -f /tmp/mujoco.zip

# Set library load path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

# Copy the mujoco license key
COPY mjkey.txt /root/.mujoco/mjkey.txt

# Install gym with mujoco
RUN pip3 install gym[mujoco]

# Test the python packages
RUN python3 -c "import gym; gym.make('Humanoid-v2'); print('done')"