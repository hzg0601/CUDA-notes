安装官网标准教程（未成功，可能仅针对9.0.0有效）

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

安装opencv教程

https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html(编译失败)

```
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
# Create build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.x
# Build
cmake --build .
```

基于opencv_contrib（编译失败）

```
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
cmake --build .

```

https://bbs.aw-ol.com/topic/227/%E5%9C%A8ubuntu%E4%B8%AD%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91opencv-4-5-1-%E8%BF%90%E8%A1%8C%E4%BA%8Etina-linux%E4%B8%AD-%E6%95%B4%E5%90%88%E5%B8%96

安装cmake-gpu后不勾选编译出错的库，可能的库dnn,stereo,....

```
sudo apt install cmake-qt-gui
```

如果pkgconfig下没有opencv.pc，则需要手动新建

https://blog.csdn.net/PecoHe/article/details/97476135
