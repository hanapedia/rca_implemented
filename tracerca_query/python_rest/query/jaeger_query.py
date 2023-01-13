from typing import List
import requests
from dataclasses import asdict
from collections import defaultdict

from query.models import ParsedJaegerTraces, ServicesQueryResponse, TracesQueryParams, TracesQueryResponse, TracesQuerySpansReduced

class JaegerQuery:
    """A class with set of queries to retrieve traces from Jaeger query service"""
    def __init__(self, host: str, port: str) -> None:
        self.url = f"http://{host}:{port}"

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

    def parse_jaeger_traces(self, tracesQueryParams: TracesQueryParams) -> List[ParsedJaegerTraces]:
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

        trace_id = trace["traceID"]
        spans = trace["spans"]
        processes = trace["processes"]

        span_map = defaultdict(list)
        for span in spans:
            span_map[span["operationName"]].append(TracesQuerySpansReduced(
                                                       startTime=span["startTime"],
                                                       duration=span["startTime"],
                                                       logs=span["logs"],
                                                       tags=span["tags"],
                                                       processID=span["processID"],
                                                   ))

        for span_pair in span_map.values():
            if len(span_pair) != 2:
                continue


        return ParsedJaegerTraces(trace_id,)
