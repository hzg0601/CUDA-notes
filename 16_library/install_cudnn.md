官网标准教程（未成功，可能仅针对9.0.0有效）
```
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.3/local_installers/12.x/cudnn-local-repo-ubuntu1804-8.9.3.28_1.0-1_amd64.deb/;
sudo dpkg -i cudnn-local-repo-ubuntu1804-8.9.3.28_1.0-1_amd64.deb;
sudo cp /var/cudnn-local-repo-ubuntu1804-8.9.3.28/cudnn-*-keyring.gpg /usr/share/keyrings/;
sudo apt-get update;
sudo apt-get -y install cudnn;
```
tar包方式安装
```
tar -xf cudnn-linux-x86_64-8.9.3.28_cuda12-archive.tar.xz
sudo cp  cudnn-linux-x86_64-8.9.3.28_cuda12-archive/include/* /usr/local/cuda-11.8/include
sudo cp  cudnn-linux-x86_64-8.9.3.28_cuda12-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod 777 /usr/local/cuda/include/cudnn*
sudo chmod 777 /usr/local/cuda/lib64/libcudnn*
```
