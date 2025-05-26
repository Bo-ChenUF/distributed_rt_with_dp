FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# port,  remote port in the container
# for jupyter, needs to be passed with -p xxxx:8888 at run time
EXPOSE 8888

###############
##### preconfig
###############
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN export DEBIAN_FRONTEND

##############################
##### key to use nvidia ubuntu
##############################
RUN DEBIAN_FRONTEND=noninteractive apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

###########################################################
##### install secure communication(libssl), python, and pip
###########################################################
RUN DEBIAN_FRONTEND=noninteractive apt update -y \
    && apt install -y \
    libssl-dev \
    python3-dev \
    python3-pip \
    && apt-get clean

##############################
##### remove temporal packages
##############################
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

#################
##### upgrade pip
#################
RUN pip3 install --no-cache-dir --upgrade pip

#################################
##### Prepare working environment
#################################
RUN mkdir /tp
WORKDIR /tp

###########################
###### Install dependencies
###########################
COPY resources/apt.txt /tp/
RUN apt update && cat apt.txt | xargs apt install -y

# install nodejs
RUN curl -sL https://deb.nodesource.com/setup_15.x | bash - && apt install -y nodejs

#############################
##### Install python packages
#############################
RUN pip install -U setuptools pip	# for cupy
COPY resources/requirements.txt /tp/
RUN pip install -r requirements.txt --no-cache-dir

####################################
##### Install and config jupyter lab
####################################
RUN pip install jupyterlab "ipywidgets>=7.5"
RUN pip install --upgrade notebook==6.4.12
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable --py widgetsnbextension

# needed otherwise bug when opening terminal from jupyter.
ENV SHELL=/bin/bash

# install the GUI backend tk
RUN apt-get update && \
      apt-get -y install sudo
#RUN sudo apt-get install python3.8-tk

#######################
##### add new sudo user
#######################
# RUN apt install sudo
ENV USERNAME python
ENV GROUPNAME python

ENV UID 5190
ENV GID 5000

RUN groupadd --gid $GID $GROUPNAME && \
     useradd -g $GROUPNAME -u $UID -m $USERNAME && \
     echo "$USERNAME:$USERNAME" | chpasswd && \
     usermod --shell /bin/bash $USERNAME && \
     usermod -aG sudo $USERNAME && \
     echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
     chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

########## docker behavior ##########
#ENTRYPOINT ["python"]
CMD /bin/bash
