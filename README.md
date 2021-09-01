# InnoTIS
InnoTIS 是 Innodisk and Aetina 用來提供 Aetina Server 運行AI模型的效果，我們結合了 NVIDIA Triton Inference Server 的技術讓使用者可以透過gRPC的方式傳送資料到我們的 Aetina Server 進行 AI 推論進而取得辨識結果。

## Ubuntu , Windows 10 通用版
**Make sure Docker is installed in client device...**

1. 下載並執行 docker

-t : allocate a pseude-TTY
-p : public container's port 5000 to host's port 5000

```
docker run -t -p 5000:5000 -t maxchanginnodisk/innotis
```

2. 開啟瀏覽器 輸入網址
```
localhost:5000
```

4. 盡情的遊玩吧！

## Jetson
>Under construction

...
