from dataclasses import dataclass
from types import NoneType
from typing import Dict, List
from collections import namedtuple

"""namedtuple for source and target services pair"""
S_T = namedtuple("S_T", ["source", "target"])

def map_status_code_grpc_to_http(grpc_status_code: int) -> int:
    """Maps grpc status code to closest http status code"""
    mapping = {
        0: 200, 
        1: 499, 
        2: 500, 
        3: 400, 
        4: 504, 
        5: 404, 
        6: 409, 
        7: 403, 
        8: 429, 
        9: 400, 
        10: 409, 
        11: 400, 
        12: 501, 
        13: 500, 
        14: 503, 
        15: 500, 
        16: 401, 
    }

    return mapping[grpc_status_code]

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
class KeyTypeValue:
    key: str
    type: str
    value: str

@dataclass
class TracesQueryGrpcLogs:
    fields: List[KeyTypeValue]
    timestamp: int

@dataclass
class ProcessesObject:
    serviceName: str
    tags: List[KeyTypeValue]

"""Processes type in trace query response"""
TracesQueryProcesses = Dict[str, ProcessesObject]

@dataclass
class TracesQuerySpans:
    traceID: str
    spanId: str
    operationName: str
    references: List[str]
    startTime: int
    duration: int
    tags: List[KeyTypeValue]
    logs: List[TracesQueryGrpcLogs]
    processID: str
    warnings: NoneType

@dataclass
class TracesQueryResponse:
    """Dataclass for FindTraces reponse data"""
    traceID: str
    spans: List[TracesQuerySpans]
    processes: Dict[str,TracesQueryProcesses] 
    warnings: NoneType

@dataclass
class TracesQuerySpansReduced:
    startTime: int
    duration: int
    tags: List[KeyTypeValue]
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
    label: int = 0

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
    label = 0
