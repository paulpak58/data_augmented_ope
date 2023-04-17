FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg-agent lsb-release software-properties-common && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get install -y software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && apt-get install -y python3-pip curl unzip \
    libosmesa-dev libglew-dev patchelf libglfw3-dev
RUN apt-get update && apt-get install -y git vim curl zsh sudo wget

# Add new user
ARG USERNAME=paulpak
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USER_PASSWORD=mujoco-pak
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME:$USER_PASSWORD" | chpasswd \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# User base dir permissions
RUN usermod -aG sudo $USERNAME
RUN chmod g+w /root

# Set user permissions and environment variables
USER $USERNAME
RUN chown $USERNAME:$USERNAME /home/$USERNAME && \
    chmod 755 /home/$USERNAME
ENV LD_LIBRARY_PATH /home/$USERNAME/.mujoco/mjpro150/bin
ENV MUJOCO_PY_MUJOCO_PATH /home/$USERNAME/.mujoco
ENV MUJOCO_PY_MJKEY_PATH /home/$USERNAME/.mujoco/mjkey.txt
ENV D4RL_DATASET_DIR /home/$USERNAME/.d4rl

# Retrieve mujoco and key
RUN curl https://www.roboti.us/download/mjpro150_linux.zip --output /tmp/mujoco.zip && \
    mkdir -p /home/$USERNAME/.mujoco && \
    unzip /tmp/mujoco.zip -d /home/$USERNAME/.mujoco && \
    rm -f /tmp/mujoco.zip
RUN curl https://www.roboti.us/file/mjkey.txt --output /home/$USERNAME/.mujoco/mjkey.txt

# D4RL and mujoco requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN git clone https://github.com/Farama-Foundation/d4rl.git /home/$USERNAME/d4rl && \
    cd /home/$USERNAME/d4rl && \
    pip3 install -e . && \
    chown $USERNAME:$USERNAME /home/$USERNAME/d4rl && \
    chmod 755 /home/$USERNAME/d4rl
ENV D4RL_SUPPRESS_IMPORT_ERROR=1

 # Add to path
RUN echo 'export PATH="$PATH:/home/$USER/"' >> /home/$USERNAME/.bashrc
RUN echo 'export PATH="$PATH:/home/$USER/.local/bin"' >> /home/$USERNAME/.bashrc
RUN echo 'export PATH="$PATH:/home/$USER/d4rl"' >> /home/$USERNAME/.bashrc
RUN echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.mujoco/mjpro150/bin"' >> /home/$USERNAME/.bashrc
RUN echo 'export MUJOCO_PY_MUJOCO_PATH="/home/$USER/.mujoco"' >> /home/$USERNAME/.bashrc
RUN echo 'export MUJOCO_PY_MJKEY_PATH="/home/$USER/.mujoco/mjkey.txt"' >>  /home/$USERNAME/.bashrc
RUN echo 'export D4RL_DATASET_DIR="/home/$USER/.d4rl"' >> /home/$USERNAME/.bashrc
RUN echo 'source ~/.bashrc' >> /home/$USERNAME/.bash_profile

# Test the python packages
#RUN python3 -c "import gym; gym.make('maze2d-umaze-v1'); print('done')"
#RUN python3 -c "import gym; gym.make('maze2d-medium-v1'); print('done')"
#RUN python3 -c "import gym; gym.make('maze2d-large-v1'); print('done')"
#RUN python3 -c "import gym; gym.make('antmaze-umaze-v1 '); print('done')"
#RUN python3 -c "import gym; gym.make('antmaze-medium-play-v1'); print('done')"
#RUN python3 -c "import gym; gym.make('antmaze-large-play-v1'); print('done')"
RUN python3 -c "import gym; gym.make('Humanoid-v2'); print('done')"