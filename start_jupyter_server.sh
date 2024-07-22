dport=8889
duser=python
dpath=$PWD/docker
gpus="--gpus all"
dimage=bchen351/distributed_r0
dname=distributed_r0
dcmd="jupyter lab --no-browser --ip 0.0.0.0"

docker run --rm -it -p $dport:8888  \
-v "$(pwd)":/home/$duser/shared \
-v $dpath/resources/jupyter:/home/$duser/.jupyter \
-v $dpath/resources/bashrc:/home/$duser/.bashrc \
-w /home/$duser/shared \
$gpus \
-u $duser -h $dname --name $dname  \
$dimage $dcmd
