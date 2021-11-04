# InnoTIS
InnoTIS 是 Innodisk and Aetina 用來提供 Aetina Server 運行AI模型的效果，我們結合了 NVIDIA Triton Inference Server 的技術讓使用者可以透過gRPC的方式傳送資料到我們的 Aetina Server 進行 AI 推論進而取得辨識結果。

---

## Ubuntu , Windows 10 通用版
**Make sure Docker is installed in client device...**

1. 運行 Docker Container
   1. 從 Dockerhub 直接下載 & 運行
       ```
       docker run -t -p 5000:5000 -t maxchanginnodisk/innotis
       ```
       * -t -> allocate a pseude-TTY
       * -p -> public container's port 5000 to host's port 5000

   2. 或者可以使用 Dockerfile 建構

        ```bash
        # Download
        $ git clone https://github.com/MaxChangInnodisk/innotis-client.git && cd innotis-client/docker
        # Build Docker Image
        $ ./build.sh
        # Check Images ( innotis:latest )
        $ docker images
        # Run Sample
        $ ./run.sh
        ```
2. 開啟瀏覽器 輸入 localhost:5000

    * 注意：Server IP 記得要修改。

3. 盡情遊玩

---

## Anaconda

1. 安裝 `anaconda` 或是 `miniconda`
2. 下載此 Github
   ```bash
   git clone https://github.com/MaxChangInnodisk/innotis-client.git && cd innotis-client
   ```
3. 安裝相依套件
   ```bash
   $ sudo apt-get update -q
   $ sudo apt-get install -qy libgl1-mesa-glx ffmpeg x264 libx264-dev
   ```
4. 建立虛擬環境
   ```bash
   conda env create -f ./docker/environment.yml
   ```
5. 開啟環境 (Optional)
   ```
   conda activate innotis

   # exit
   # conda deactivate
   ```
6. 執行程式
   ```bash
   python3 app.py
   ```

---

## Jetson
>Under construction

...