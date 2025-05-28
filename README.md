# Scalable Distributed Reproduction Numbers of Network Epidemics with Differential Privacy
Please find the paper: https://arxiv.org/abs/2501.18862.

To run the experiments, please go to the notebooks folder and run all notebooks there in order.

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

## Updated version for OJCSYS
Note that the transmission network B used from [17] has the diagonal entries all set to one. In our work, we only constructed distributed reproduction numbers between the nodes and clusters, avoiding the use of the diagonal entries of the transmission matrix B from [17]. For constructing the network-level reproduction numbers, for each ${i}$-th row, we replaced the diagonal value of one with the average of the row sums. However, for the local reproduction numbers of individual nodes and clusters, this approach may result in values that do not fit the model to real data. Since parameter estimation is not the main focus of our work, we did not attempt to estimate $\beta_{ii}$ , for all ${i}$, ourselves. Consequently, we did not plot the cluster-level local reproduction numbers for the three regions in our paper.

[17] G. Le Treut, G. Huber, M. Kamb, K. Kawagoe, A. McGeever, J. Miller, R. Pnini, B. Veytsman, and D. Yllanes, “A high-resolution flux-matrix model describes the spread of diseases in a spatial network and the effect of mitigation strategies,” Sci. Rep., vol. 12, no. 1, p. 15946, 2022.
