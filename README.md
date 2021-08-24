# InnoTIS
InnoTIS 是一個可以提供 NVIDIA Triton Inference Server 技術的網頁應用，此應用可以透過 HTTP/gRPC 的方式傳送資料到Server並且在進行完推論後回傳結果到Client端。

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