# Tags cannot contain plus sign:
# https://docs.docker.com/engine/reference/commandline/tag/
TAG := $(shell git describe --tags --dirty | tr + -)

all: docker2singularity

docker_build:
	docker build --no-cache -f Dockerfile -t fastplms:$(TAG) .

docker2singularity: docker_build
	docker run -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/fastplms:/output \
		--privileged -t --rm \
		quay.io/singularity/docker2singularity \
		--name fastplms_$(TAG).sif \
		fastplms:$(TAG)