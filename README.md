# InnoTIS
InnoTIS 是一個可以提供 NVIDIA Triton Inference Server 技術的網頁應用，此應用可以透過 HTTP/gRPC 的方式傳送資料到Server並且在進行完推論後回傳結果到Client端。

## Ubuntu 

1. 下載 docker images
```
docker pull maxchanginnodisk/innotis
```

2. 執行 docker
```
docker run -p 5000:5000 -t maxchanginnodisk/innotis
```

3. 開啟瀏覽器 輸入網址
```
localhost:5000
```

4. 盡情的遊玩吧！

## Jetson
*Make Sure Your JetPack Version is >= 4.6*

1. 安裝 Triton 的相依套件
```
sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        software-properties-common \
        autoconf \
        automake \
        build-essential \
        cmake \
        git \
        libb64-dev \
        libre2-dev \
        libssl-dev \
        libtool \
        libboost-dev \
        libcurl4-openssl-dev \
        libopenblas-dev \
        rapidjson-dev \
        patchelf \
        zlib1g-dev
```

2. 更新 CMake 3.18.4 ( About 15:28~)
```
sudo apt remove cmake -y
cd ~ && wget https://cmake.org/files/v3.18/cmake-3.18.4.tar.gz
tar -xf cmake-3.18.4.tar.gz
cd cmake-3.18.4 && ./configure && sudo make install
rm -rf cmake-3.18.4
```

3. 安裝 Client 相依套件
```
sudo apt install -y --no-install-recommends \
    curl \
    libopencv-dev=3.2.0+dfsg-4ubuntu0.1 \
    libopencv-core-dev=3.2.0+dfsg-4ubuntu0.1 \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev

pip3 install --upgrade wheel setuptools cython && \
pip3 install --upgrade grpcio-tools numpy==1.19.4 future attrdict
```


1. 下載 Github
```
git clone https://github.com/MaxChangInnodisk/InnoTIS.git
cd InnoTIS
```

2. 建構虛擬環境
```
sudo apt install anaconda -y 
```

## Windows 10 