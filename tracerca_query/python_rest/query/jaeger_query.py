from typing import Dict, List, DefaultDict
import requests
from dataclasses import asdict
from collections import defaultdict

from query.models import ParsedJaegerTraces, ServicesQueryResponse, TracesQueryParams, TracesQueryProcesses, TracesQueryResponse, TracesQuerySpans, TracesQuerySpansReduced, S_T, map_status_code_grpc_to_http

class JaegerQuery:
    """A class with set of queries to retrieve traces from Jaeger query service"""
    def __init__(self, hostname: str, port: str) -> None:
        self.url = f"http://{hostname}:{port}"

    def query_services(self) -> ServicesQueryResponse:
        """
        Get services. 
        Calls endpoint: /api/services
        """

        path = "/api/services"
        query_url = self.url + path
        res = requests.get(query_url)
        if res.status_code != 200:
            raise Exception(f"status: {res.status_code}")
        res = res.json()

        return ServicesQueryResponse(services=res['data'])
            
    def query_traces(self, tracesQueryParams: TracesQueryParams) -> List[TracesQueryResponse]:
        """
        Get traces with given query parameters. 
        Calls endpoint: /api/traces
        """

        path = "/api/traces"
        query_url = self.url + path
        res = requests.get(query_url, params=asdict(tracesQueryParams))
        if res.status_code != 200:
            raise Exception(f"status: {res.status_code}")
        res = res.json()
        res: List[TracesQueryResponse] = res["data"]

        return res

    def query_and_parse_jaeger_traces(self, tracesQueryParams: TracesQueryParams) -> List[ParsedJaegerTraces]:
        """
        Queries traces and parse them

        param: tracesQueryParams: Parameters for traces query
        """

        traces = self.query_traces(tracesQueryParams)
        parsed_traces: List[ParsedJaegerTraces] = []
        for trace in traces:
            parsed_traces.append(self.parse_traces(trace))

        return parsed_traces

    def parse_traces(self, trace: TracesQueryResponse) -> ParsedJaegerTraces:
        """Parse each trace in query response"""

        trace_id: str = trace["traceID"]
        spans: List[TracesQuerySpans ] = trace["spans"]
        processes: TracesQueryProcesses = trace["processes"]

        span_map: DefaultDict[List[TracesQuerySpansReduced]] = defaultdict(list)
        for span in spans:
            span_map[span["operationName"]].append(TracesQuerySpansReduced(
                                                       startTime=span["startTime"],
                                                       duration=span["duration"],
                                                       logs=span["logs"],
                                                       tags=span["tags"],
                                                       processID=span["processID"],
                                                   ))

        span_map_parser = SpanMapParser(
            trace_id = trace_id, 
            span_map = span_map, 
            processes = processes
        )
        return span_map_parser.parse()
#         return ParsedJaegerTraces(trace_id,)
class SpanMapParser:
    def __init__(self, trace_id: str, span_map: Dict[str, List[TracesQuerySpansReduced]], processes: TracesQueryProcesses) -> None:
        self.trace_id = trace_id
        self.span_map = span_map
        self.processes = processes
        self.timestamp: List[float] = []
        self.latency: List[float] = []
        self.http_status: List[float] = []
        self.endtime: List[float] = []
        self.s_t: List[S_T] = []

    def parse(self) -> ParsedJaegerTraces:
        """parse into ParsedJaegerTraces"""
        for span_pair in self.span_map.values():
            if len(span_pair) != 2:
                continue
            span_pair = self.ensure_pair_order(span_pair)
            self.timestamp.append(self.get_timestamps(span_pair))
            self.latency.append(self.get_latency(span_pair))
            self.http_status.append(self.get_http_status(span_pair))
            self.endtime.append(self.get_endtime(span_pair))
            self.s_t.append(self.get_s_t(span_pair))
        
        return ParsedJaegerTraces(
            trace_id = self.trace_id,
            timestamp = self.timestamp,
            latency = self.latency,
            http_status = self.http_status,
            endtime = self.endtime,
            s_t = self.s_t,
        )

    def ensure_pair_order(self, span_pair: List[TracesQuerySpansReduced]) -> List[TracesQuerySpansReduced]:
        """Ensure that the spans are ordered correctly"""
        if span_pair[0].startTime > span_pair[1].startTime:
            span_pair.reverse()
        return span_pair
    
    def get_timestamps(self, span_pair: List[TracesQuerySpansReduced]) -> float:
        """Use the startTime of the older span, which should be at index 0"""
        return float(span_pair[0].startTime)

    def get_latency(self, span_pair: List[TracesQuerySpansReduced]) -> float:
        """
        UNARY GRPC SPANS ONLY 
        UNTESTED ON STREAMS

        take the difference between the timestamp of SEND and RECV
        """
        if len(span_pair[0].logs) == 0:
            raise Exception("get latency only supports unary grpc spans at the moment")

        latency = span_pair[0].logs[1]["timestamp"] - span_pair[0].logs[0]["timestamp"]
        return float(latency)

    def get_http_status(self, span_pair: List[TracesQuerySpansReduced]) -> float:
        """Returns grpc status code mapped http status code"""
        tags = span_pair[0].tags
        grpc_status_code = 2
        for tag in tags:
            if tag["key"] == "rpc.grpc.status_code":
                grpc_status_code = tag["value"]
        return float(map_status_code_grpc_to_http(grpc_status_code))

    def get_endtime(self, span_pair: List[TracesQuerySpansReduced]) -> float:
        """Returns the end time of older span"""
        return float(span_pair[0].startTime + span_pair[0].duration)

    def get_s_t(self, span_pair: List[TracesQuerySpansReduced]) -> S_T:
        """Returns the service name pair"""
        source = self.processes[span_pair[0].processID]["serviceName"]
        target = self.processes[span_pair[1].processID]["serviceName"]
        return (source, target)

