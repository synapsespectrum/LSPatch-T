IMAGE := TSPytorch
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	--gpus all \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init:
	docker build -t ${IMAGE} .

dataset:
	mkdir -p dataset/ETT && \
		wget -O dataset/ETT/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv && \
		wget -O dataset/ETT/ETTh2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv && \
		wget -O dataset/ETT/ETTm1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv && \
		wget -O dataset/ETT/ETTm2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv && \
		wget -O dataset/ECL.csv "https://drive.google.com/uc?export=download&id=1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP" && \
		wget -O dataset/WTH.csv "https://drive.google.com/uc?export=download&id=1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG"


run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		${IMAGE} ${module}

bash_docker:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}

.require-module:
ifndef module
	$(error module is required)
endif
