# Instruction for develoying Tensorflow Serving on Centos 7

### Install Python3.8 on CentOS7

```bash
yum -y groupinstall "Development tools"
yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel screen nano vim
yum install libffi-devel -y
wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tgz
tar -zxvf  Python-3.8.3.tgz
mkdir /usr/local/python3
cd Python-3.8.3
./configure --prefix=/usr/local/python3
make && make install
ln -s /usr/local/python3/bin/python3 /usr/bin/python3
ln -s /usr/local/python3/bin/pip3 /usr/bin/pip3
pip3 install keras-preprocessing flask requests numpy opencv-python requests
```

### Start Tensorflow Serving

```bash
yum install docker screen
systemctl start docker
docker pull tensorflow/serving
screen
docker run -p 8501:8501 --mount type=bind,source=/root/mymodel/,target=/models/my_model/1/ -e MODEL_NAME=my_model -t tensorflow/serving
```

### End Tensorflow Serving

- ```Ctrl+C```
- ```Ctrl+Z```
- ```systemctl stop docker```
- ```use top command to find PID and kill```

### Start Flask Serving

```bash
cd serve
screen
python3/python3.8 App.py
```

### Stop Flask Serving

- ```Ctrl+C```
- ```Ctrl+Z```
- ```use top command to find PID and kill```


## Adanced Points

- TensorFlow Serving can cooperate with Kubernetes, which can deploy in many serves at the same time, it is more suitable for companies to deploy

### Python3.8 Installtion Reference

 - https://blog.csdn.net/fanxl10/article/details/106854062