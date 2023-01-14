package main

import (
	"context"
	// "reflect"
	// "unsafe"

	// "encoding/json"
	"io"
	"log"
	// "time"

	types "github.com/gogo/protobuf/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/hanapedia/rca_methods/tracerca_query/go/api_v3"
)

func main(){
	var optsClient []grpc.DialOption
  optsClient = append(optsClient, grpc.WithTransportCredentials(insecure.NewCredentials()))

  conn, err := grpc.DialContext(context.Background(),"localhost:16685", optsClient...)
	if err != nil {
		log.Fatalf("Cannot establish connection with the server: %v", err)
	}
	defer conn.Close()

  queryServiceClient := pb.NewQueryServiceClient(conn)

  gsreq := pb.GetServicesRequest {}

  gsres, err := queryServiceClient.GetServices(context.Background(), &gsreq)
  if err != nil {
    log.Fatalln("Failed to retrieve services")
  }
  log.Printf("%v", gsres)

  findTraces(queryServiceClient)
  // getTrace(queryServiceClient)
}

type SpansResponseChunkParser struct {
  Result pb.SpansResponseChunk `json:"result,omitempty"`
}

func getTrace(client pb.QueryServiceClient) {
  traceRequestParam := pb.GetTraceRequest{
    TraceId: "0c236127a3dc2c6edd44463fb1c14271",
  }
  log.Println(traceRequestParam)
  stream, err := client.GetTrace(context.Background(), &traceRequestParam)
	if err != nil {
    log.Fatalf("query failed: %v", err)
	}
  for {
      feature, err := stream.Recv()
      if err == io.EOF {
          break
      }
      if err != nil {
          log.Fatalf("%v", err)
      }
      // var srcParser *SpansResponseChunkParser
      // json.Unmarshal([]byte(feature.String()), srcParser)
      log.Printf("%#v", feature.ResourceSpans[0].ScopeSpans[0].Spans[0].String()) // unmarshall
      // rv := reflect.ValueOf(feature).Elem()
      // rf := rv.FieldByName("results")
      // rf = reflect.NewAt(rf.Type(), unsafe.Pointer(rf.UnsafeAddr())).Elem()
      // log.Println(rf.Interface())
  }
}

func findTraces(client pb.QueryServiceClient) {
  // startTimeMax := types.Timestamp {
  // now := time.Now().Unix()
  // startTimeMax := types.Timestamp {
  //   Seconds: 1673534532,
  //   Nanos: 1e8,
  // }
  // startTimeMin := types.Timestamp {
  //   Seconds: 1673530932,
  //   Nanos: 1e8,
  // }
  startTimeMax := types.TimestampNow()
  startTimeMin := types.Timestamp {
    Seconds: startTimeMax.GetSeconds() - 3600,
    Nanos: startTimeMax.Nanos,
  }
  // maxDuration := types.Duration {
  //   Seconds: 1,
  //   Nanos: 0,
  // }
  // minDuration := types.Duration {
  //   Seconds: 0,
  //   Nanos: 1e5,
  // }
  
  traceQueryParameters := pb.TraceQueryParameters{
    ServiceName: "gateway",
    OperationName: "all",
    StartTimeMin: &startTimeMin,
    StartTimeMax: startTimeMax,
    // DurationMin: &minDuration,
    // DurationMax: &maxDuration,
    NumTraces: 20,
  }
  stream, err := client.FindTraces(context.Background(), &pb.FindTracesRequest{Query: &traceQueryParameters})
	if err != nil {
    log.Fatalf("query failed: %v", err)
	}
  i := 0
  for {
      feature, err := stream.Recv()
      if err == io.EOF {
          log.Printf("eof reached after %v", i)
          break
      }
      if err != nil {
          log.Fatalf("%v", err)
      }
      i++
      log.Println(feature)
  }
}

