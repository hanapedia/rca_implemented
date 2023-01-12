#!/bin/bash
# docker run --rm -v${PWD}:${PWD} -w${PWD} rvolosatovs/protoc --proto_path=./api_v2 --python_out=./python_gen model.proto
docker run --rm -v${PWD}:${PWD} -w${PWD} rvolosatovs/protoc --proto_path=./api_v2 --python_out=./python_gen query.proto --grpc-python_out=./python_gen
# docker run --rm -v${PWD}:${PWD} -w${PWD} rvolosatovs/protoc --proto_path=./api_v2 --python_out=./python_gen collector.proto
# docker run --rm -v${PWD}:${PWD} -w${PWD} rvolosatovs/protoc --proto_path=./api_v2 --python_out=./python_gen sampling.proto
