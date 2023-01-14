package main_test

import (
	// "encoding/hex"
	// "encoding/json"
	"log"
	"testing"
	"time"

	// v1 "go.opentelemetry.io/proto/otlp/trace/v1"

	pb "github.com/hanapedia/rca_methods/tracerca_query/go/api_v3"
)
//
// func TestEncoding(t *testing.T) {
//   str_in :=  "0c236127a3dc2c6edd44463fb1c14271"
//   encoded := hex.EncodeToString([]byte(str_in))
//   if encoded == "" {
//     t.Fail()
//   }
//
//   // log.Printf("%v", encoded)
//
// }

type SpansResponseChunkParser struct {
  Result pb.SpansResponseChunk `json:"result,omitempty"`
}
//
// func TestResponseChunkParsing(t *testing.T) {
//   var rs [] *v1.ResourceSpans 
//   spc := pb.SpansResponseChunk {
//     ResourceSpans: rs,
//   }
//   // res := SpansResponseChunkParser {
//   //   Result: spc,
//   // }
//
//   var parsed *SpansResponseChunkParser
//   json.Unmarshal([]byte(spc.String()), parsed)
//   log.Printf("%v", parsed)
// }

func TestAssertTime(t *testing.T) {
  time_dif := time.Now().Unix() - 1673534532
  log.Println(time_dif)
  t.Fail()
}
