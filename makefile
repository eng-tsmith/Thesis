help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.5
CUDA_VERSION?=8.0
CUDNN_VERSION?=6
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
	docker build -t keras --build-arg python_version=3.5 --build-arg cuda_version=8.0 --build-arg cudnn_version=6 -f Dockerfile /.
	docker build -t keras --build-arg python_version=3.5 --build-arg cuda_version=8.0 --build-arg cudnn_version=6 -f Dockerfile ./

bash: 
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras bash
	GPU=0 nvidia-docker run -it -v /cygdrive/c/Users Smith/Desktop:/src/workspace -v "/home/Timothy Smith/Data":/data --env KERAS_BACKEND=tensorflow keras bash
