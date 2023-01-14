import os
from pathlib import Path
from typing import List
import click
import time
from dataclasses import asdict

from query.jaeger_query import JaegerQuery
from query.models import ParsedJaegerTraces, TracesQueryParams
from query._utils import load_pickle_file, write_pickle_file

SEC_TO_MICROSEC = 1_000_000

@click.command("Query traces from jaeger and prometheus, and TraceRCA input format")
@click.option("-o", "--output_path", "output_path", type=str, help="Output path for pickle file containing formatted traces", required=True)
@click.option("-a", "--output_append", "output_append", is_flag=True, help="Append if the outfile exits")
@click.option("-t", "--timestamp", "timestamp", type=int, default=-1, help="From when to look back to look for traces in Unix seconds")
@click.option("-l", "--loopback", "loopback", type=int, default=60, help="For how long to loopback to look for traces, in seconds")
@click.option("-r", "--trace_sampling_rate", "trace_sampling_rate", type=int, default=4, help="Trace sampling rate in seconds")
@click.option("-g", "--gateway_name", "gateway_name", type=str, default="gateway", help="The name of the gateway service")
@click.option("-h", "--hostname", "hostname", type=str, default="localhost", help="hostname for jaeger query")
@click.option("-p", "--port", "port", type=str, default="16686", help="port for jaeger query")
@click.option("-p", "--prometheus", "include_prometheus", is_flag=True, help="flag to indicate whether or not to include pormetheus metrics")
def main(
        output_path,
        output_append,
        timestamp,
        loopback, 
        trace_sampling_rate, 
        gateway_name, 
        hostname, 
        port, 
        include_prometheus
        ):

    if include_prometheus:
        raise Exception("Currently unsupported")

    if timestamp <= 0:
        timestamp = int(time.time())

    start = timestamp - loopback
    limit = loopback * trace_sampling_rate

    params = {
        "start": start * SEC_TO_MICROSEC,
        "end": timestamp * SEC_TO_MICROSEC,
        "loop": "custom",
        "limit": limit,
        "service": gateway_name, 
    }
    tracesQueryParams = TracesQueryParams(**params)
    jaegerQuery = JaegerQuery(hostname=hostname, port=port)
    traces = jaegerQuery.query_and_parse_jaeger_traces(tracesQueryParams)
    traces = list(map(lambda x: asdict(x), traces))
    
    output_path = Path(output_path)
    if output_append:
        original_traces = load_traces_pickle(output_path)
        if not isinstance(original_traces, list):
            raise Exception("File already exists but does not match the format wanted. (List[ParsedJaegerTraces])")
        
        traces = original_traces.extend(traces)
    
    write_pickle_file(output_path, traces)

if __name__ == "__main__":
    main()
