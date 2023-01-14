import grpc
from proto_gen_python.api_v3.query_service_pb2_grpc import * 
from pprint import pprint

def main():
    channel = grpc.insecure_channel("localhost:16685")
    stub = QueryServiceStub(channel)
    # stub.FindTraces({
    #                     "service_name": "gateway",
    #                     "operation_name": "all",
    #                     "tags": {},
    #                     "start_time_min": 1670300574256000,
    #                     "start_time_max": 1670300574257000,
    #                     "duration_min": '',
    #                     "duration_max": '',
    #                     "search_depth": '',
    #                 })
    


    svc = stub.GetServices({})
    # stub.GetTrace(request={"trace_id":bytes("d4741835c98b39a12a9bccf6a70e51ea", "utf-8")})
    pprint(svc)

if __name__ == "__main__":
    main()
