# Build InnoTIS's Docker Image

---

1. Build from source

	* Get Repository
	```
	git clone https://github.com/MaxChangInnodisk/InnoTIS.git
	```

	* Move to docker directory 
	```
	cd InnoTIS/docker
	```

	* Build Docker Image
	```
	./build.sh
	```

	* Check
	```
	docker images
	```	
	You will get the result like:
	```
	REPOSITORY	TAG		IMAGE ID       CREATED          SIZE
	innotis		latest          7df4db57614a   32 seconds ago   1.31GB
	```

	* Run 
	```
	./run.sh
	```

2. Pull from docker hub

	* Run docker command
	```
	
	```
