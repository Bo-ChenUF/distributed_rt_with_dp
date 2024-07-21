# [The name of our paper]
We will document the step we took to run the simulations. We may make another readme when this repository is public.

## Prerequisites
You can build a working environment using Docker.
First, please follow the instructions to install Docker (see the [doc](https://docs.docker.com/get-docker/)).
Add an Ubuntu 20.04 subsystem for you computer if you do not have one.
For Windows user, open PowerShell or Windows Command Prompt as an administrator, enter the command 

```
wsl --install -d ubuntu-20.04
```

, and restart the computer. 

Next, open and edit the Dockerfile located at `docker/Dockerfile`. 
Change the values of `UID` and `GID` variables to match your own values.
Save and close the file.
Build the Docker image by executing (at the root of the repository):

```
docker build -t <yourname>/distributed_r0 docker/.
```
Note that this Docker image is based on nvidia/cuda:12.4.1-devel-ubuntu20.04. 
If this is not compatible with your system, you will need to use a different Debian image (eg ubuntu), remove the cupy-cuda110 library in the docker/resources/requirements.txt file, and modify the scripts using CuPy (eg use NumPy instead).


