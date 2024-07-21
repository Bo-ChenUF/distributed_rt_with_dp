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

## Running the notebooks in a Jupyter server
Once the image has been built, edit the `start_jupyter_server.sh` and change the value of the variable `dimage` to match the name of the Docker image created in the last step. 
If you are not using a GPU (see previous section) set the value of the variable `gpus` to "". Execute in a terminal (at the root of the repository):

```
bash start_jupyter_server.sh
```

Copy the prompted token and paste it in the corresponding field when entering the jupyter server. 
The jupyter server can be accessed by entering `http://localhost:8889` in a web browser.

## Connect running containers to VSCode
To use a docker contain as a development environment, add the following extensions to your VSCode:

1) docker:
2) Dev Container
3) SSH-remote
4) WSL
