# Notes from trying to request the grpc api_v3
## Things tried, and why they failed
0. tried using api_v2 as it was listed stable in the Jaeger document
  - turns out it is not and fails to query on some endpoints
  - moved on to using v3
1. Use Python to call grpc endpoints
  - struggle with python module imports
    - invalid venv config. fixed by renewing venv
  - difficult without static typing
  - python protobuf used to generate the grpc code were out of date for quite some time
  - moved on to using go
2. Use go to call grpc endpoints
  - used grpcurl to test reflection
    - reflection for Jaeger grpc api is limited and only allows listing of the endpoints
    - did not reveal much information
  - `GetServices` worked right away
  - `GetTrace` had two steps of issues
    - error parsing trace id. `strconv.UInt` invalid syntax
      - turns out I was passing empty struct
      - tried messing around with hex encoding for no reason
    - error printing the returned stream
      - reflect.ValueOf.Interface error, which indicates that there unexported fields
      - thought that the documentation for the endpoint in Jaeger, saying that envelope needs to be stripped off, was the cause. but was not, it was just the issue with trying to print structs
      - fixed with printf("%#v", res)
  - `FindTraces` did not return error but did not return any traces either 
    - tried playing around with timestamps, but no luck
    - the reason why it doesn't work is still unknown
    - decided to move on to using the REST api integration of api_v3 via grpc gateway
  - the remark on jaeger documentation only applied to when you are using grpc gateway. the documentation was there for the use with query api used by the dashboard
    - turns out that the rest api of jaeger query is using api_v3 with grpc gateway
