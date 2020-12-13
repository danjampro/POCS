#!/bin/bash
set -eu

docker build \
  --build-arg "image_url=gcr.io/panoptes-exp/panoptes-utils:develop" \
  -t "huntsmanarray/panoptes-pocs:commissioning" \
  -f "${POCS}/docker/Dockerfile" \
  "${POCS}"

docker push huntsmanarray/panoptes-pocs:commissioning
