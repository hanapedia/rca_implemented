import requests
import time
import argparse


def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    # parser.add_argument('--folder', type=str, required=False,
    #                     default='1',
    #                     help='folder name to store csv file')
    
    parser.add_argument('--length', type=int, required=False,
                    default=150,
                    help='length of time series')

    parser.add_argument('--step', type=str, required=False,
                    default='14s',
                    help='length of step')
    # parser.add_argument('--url', type=str, required=False,
    #                 default='http://localhost:9090/api/v1/query',
    #                 help='url of prometheus query')

    return parser.parse_args()

if __name__ == '__main__':
    metric_step = '14s'
    args = parse_args()

    # folder = args.folder
    len_second = args.length
    prom_url_range = 'http://localhost:31090/api/v1/query_range'
    prom_url = 'http://localhost:31090/api/v1/query'

    end_time = time.time()
    # end_time = 1662556769.906268,
    start_time = end_time - len_second
    # start_time = "2022-09-07 10:14:05"
    # end_time = "2022-09-07 10:24:05"
    latency_queries = [
        # Added conditions to filter out unkown services(external requests)
        # captures requests
        # median of increase rate in request duration
        'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter="source", destination_workload_namespace="sock-shop", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload, le))',
        # captures data transfer involving databases
        'sum(irate(istio_tcp_sent_bytes_total{reporter="source", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload) / 1000',
        'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="sock-shop", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload, le))',
        'sum(irate(istio_tcp_sent_bytes_total{reporter="destination", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload) / 1000',
    ]

    # Requires string formatting with pods and nodes names

    pod_name = "user-7d4f648858-2k492"
    instance = "192.168.100.36:9100"
    metrics_query = [
        # retrieve cpu usage of each micorservice
        # attribute container_name replaced by container
        # attribute pod_name replaced by pod
        # 'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", container_name!~"POD|istio-proxy|"}[1m])) by (pod_name, instance, container)',
        'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", container!~"POD|istio-proxy|"}[1m])) by (pod, instance, container)' ,

        # retrieve network transfered packets of a pod
        'sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod="%s"}[1m])) / 1000 * sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod="%s"}[1m])) / 1000' % (pod_name, pod_name),

        # retrieve memory metrics of a pod
        'sum(rate(container_memory_working_set_bytes{namespace="sock-shop", pod="%s"}[1m])) / 1000' % pod_name,

        # retrieve network transmission speed of a nodes
        # query node_network_transmit_packets replaced by node_network_transmit_packets_total
        # device eth0 replaced by ens3 as the nodes are ubuntu instance
        # 'rate(node_network_transmit_packets{device="eth0", instance="%s"}[1m]) / 1000',
        'rate(node_network_transmit_packets_total{device="ens3", instance="%s"}[1m]) / 1000' % instance,

        # retrieve cpu metrics of a node
        # query node_cpu replaced by node_cpu_seconds_total
        # 'sum(rate(node_cpu{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="%s" }[1m])) / count(node_cpu{mode="system", instance="%s"})',
        'sum(rate(node_cpu_seconds_total{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="%s" }[1m])) / count(node_cpu_seconds_total{mode="system", instance="%s"})' % (instance, instance),

        # retrieve memory metrics of a node
        # query node_memory_MemAvailable replaced by node_memory_MemAvailable_bytes
        # query node_memory_MemTotal replaced by node_memory_MemTotal_bytes
        # '1 - sum(node_memory_MemAvailable{instance="%s"}) / sum(node_memory_MemTotal{instance="%s"})'
        '1 - sum(node_memory_MemAvailable_bytes{instance="%s"}) / sum(node_memory_MemTotal_bytes{instance="%s"})' % (instance, instance)
    ]

    # Must be used with regular query url
    graph_queries = [
        # Added conditions to filter out unkown services(external requests)
        'sum(istio_tcp_received_bytes_total{destination_workload!="unknown", source_workload!="unknown"}) by (source_workload, destination_workload)',
        'sum(istio_requests_total{destination_workload_namespace="sock-shop", destination_workload!="unknown", source_workload!="unknown"}) by (source_workload, destination_workload)',

        # replace attribute container_name by container
        # add | behind istio-proxy to filter out results without container attributes 
        # 'sum(container_cpu_usage_seconds_total{namespace="sock-shop", container_name!~"POD|istio-proxy"}) by (instance, container)'
        'sum(container_cpu_usage_seconds_total{namespace="sock-shop", container!~"POD|istio-proxy|"}) by (instance, container)'
    ]

    for i, query in enumerate(latency_queries):
        response = requests.get(prom_url_range,
                            params={'query': query,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
        results = response.json()
        if results['data']['result'] != []:
            print("----------------------")
            print('latency query %s results' % i)
            print(results)

    for i, query in enumerate(metrics_query):
        response = requests.get(prom_url_range,
                            params={'query': query,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
        results = response.json()
        if results['data']['result'] != []:
            print("----------------------")
            print('metrics query %s results' % i)
            print(results)

    for i, query in enumerate(graph_queries):
        response = requests.get(prom_url,
                            params={'query': query})
        results = response.json()
        if results['data']['result'] != []:
            print("----------------------")
            print('graph query %s results' % i)
            print(results)

