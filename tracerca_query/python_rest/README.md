# Script for querying required data for TraceRCA
## Pipeline
1. Obtain trace data from Jaeger query
2. Format the traces data with timestamps and latency for each invocation
3. Query additional metrics from Prometheus using the invocation timestamp 
4. Save object as pickle

## Tasks
### Format the data
- extract source-target pairs for each invocations in the trace
  - only consider grpc for now
  - use operationName field of each span
  - and processID to get the host name
### Issues
- fix the invocation serialization
