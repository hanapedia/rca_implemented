from dataclasses import dataclass
from types import NoneType
from typing import Dict, List
from collections import namedtuple

"""namedtuple for source and target services pair"""
S_T = namedtuple("S_T", ["source", "target"])

@dataclass
class ServicesQueryResponse:
    services: List[str]

@dataclass
class TracesQueryParams:
    """Dataclass for FindTraces query parameter"""
    start: str 
    end: str 
    loop: str 
    limit: str 
    service: str 
    # maxDuration: str = "10s"
    # minDuration: str = "0s"

@dataclass
class TracesQuerySpansTags:
    key: str
    type: str
    value: str

@dataclass
class TracesQueryGrpcLogs:
    fields: List[TracesQuerySpansTags]
    timestamp: int

@dataclass
class TracesQuerySpans:
    traceID: str
    spanId: str
    operationName: str
    references: List[str]
    startTime: int
    duration: int
    tags: List[TracesQuerySpansTags]
    logs: List[TracesQueryGrpcLogs]
    processID: str
    warnings: NoneType

@dataclass
class TracesQueryResponse:
    """Dataclass for FindTraces reponse data"""
    traceID: str
    spans: List[TracesQuerySpans]
    processes: Dict
    warnings: NoneType

@dataclass
class TracesQuerySpansReduced:
    startTime: int
    duration: int
    tags: List[TracesQuerySpansTags]
    logs: List[TracesQueryGrpcLogs]
    processID: str

@dataclass
class ParsedJaegerTraces:
    trace_id: str
    timestamp: List[float]
    latency: List[float]
    http_status: List[float]
    endtime: List[float]
    s_t: List[S_T]

@dataclass
class TraceRcaTraces:
    trace_id: str
    timestamp: List[float]
    latency: List[float]
    http_status: List[float]
    cpu_use: List[float]
    mem_use_percent: List[float]
    mem_use_amount: List[float]
    file_write_rate: List[float]
    file_read_rate: List[float]
    net_send_rate: List[float]
    net_receive_rate: List[float]
    endtime: List[float]
    s_t: List[S_T]
    label: 0
