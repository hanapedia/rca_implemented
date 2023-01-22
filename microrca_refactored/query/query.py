import os
import pandas as pd
import networkx as nx
from pathlib import Path
import requests

from ._utils import save_as_pickle

class MicroRCAQuery:
    def __init__(self, prom_url, namespace, fault_dir, len_seconds, end_time, metric_step, node_dict) -> None:
        self.prom_url = prom_url
        self.prom_url_range = f"{prom_url}_range"
        self.namespace = namespace
        self.fault_dir = fault_dir
        self.len_seconds = len_seconds
        self.start_time = end_time - len_seconds
        self.end_time = end_time
        self.metric_step = metric_step
        self.smoothing_window = 12
        self.node_dict = node_dict
        self.tcp = False

    def query_prometheus_range(self, query: str):
        """Query prometheus range with given PromQL"""
        response = requests.get(self.prom_url_range,
                                params={'query': query,
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        return response.json()['data']['result']

    def query_prometheus(self, query: str):
        """Query prometheus with given PromQL"""
        response = requests.get(self.prom_url, params={'query': query})
        return response.json()['data']['result']

    def latency_50(self, reporter: str):
        """
        Query latency reported by either source or destination in a csv
        
        reporter can be 'source' or 'destination'
        """
        latency_df = pd.DataFrame()
        latency_query = self.get_latency_promQL(reporter)
        latencies = self.query_prometheus_range(latency_query)
        latency_df = self.map_latency_to_df(latency_df, latencies)

        if self.tcp:
            tcp_query = self.get_tcp_sent_promQL(reporter)
            tcpbytes = self.query_prometheus_range(tcp_query)
            latency_df = self.map_latency_to_df(latency_df, tcpbytes, True)

        filepath = self.fault_dir / f'latency_{reporter}_50.csv'

        if os.path.isfile(filepath):
            original_df = pd.read_csv(filepath)
            latency_df = pd.concat([original_df, latency_df], ignore_index=True)

        latency_df.set_index('timestamp')
        latency_df.to_csv(filepath, index=False)
    
    def map_latency_to_df(self, latency_df: pd.DataFrame, results: list, is_tcp: bool=False) -> pd.DataFrame:
        """Map query results to dataframe"""

        for result in results:
            dest_svc = result['metric']['destination_workload']
            src_svc = result['metric']['source_workload']
            name = src_svc + '_' + dest_svc
            values = result['values']

            values = list(zip(*values))
            if 'timestamp' not in latency_df:
                timestamp = values[0]
                latency_df['timestamp'] = timestamp
                latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            latency_df[name] = pd.Series(metric)

            if is_tcp:
                latency_df[name] = latency_df[name].astype('float64').rolling(window=self.smoothing_window, min_periods=1).mean()
            else:
                latency_df[name] = latency_df[name].astype('float64')  * 1000

        return latency_df

    def svc_metrics(self):
        """Query service metrics and save as csv"""
        ctn_cpu_query = self.get_ctn_cpu_promQL()
        ctn_cpu_use_results = self.query_prometheus_range(ctn_cpu_query)

        for result in ctn_cpu_use_results:
            df = pd.DataFrame()
            svc = result['metric']['container']
            pod_name = result['metric']['pod']
            nodename = result['metric']['instance']
            values = result['values']

            values = list(zip(*values))
            if 'timestamp' not in df:
                timestamp = values[0]
                df['timestamp'] = timestamp
                df['timestamp'] = df['timestamp'].astype('datetime64[s]')
            metric = pd.Series(values[1])
            df['ctn_cpu'] = metric
            df['ctn_cpu'] = df['ctn_cpu'].astype('float64')

            # df['ctn_network'] = self.ctn_network(pod_name)
            # df['ctn_network'] = df['ctn_network'].astype('float64')

            df['ctn_memory'] = self.ctn_memory(pod_name)
            df['ctn_memory'] = df['ctn_memory'].astype('float64')

            instance = self.node_dict[nodename]

            df_node_cpu = self.node_cpu(instance)
            df = pd.merge(df, df_node_cpu, how='left', on='timestamp')

            df_node_network = self.node_network(instance)
            df = pd.merge(df, df_node_network, how='left', on='timestamp')

            df_node_memory = self.node_memory(instance)
            df = pd.merge(df, df_node_memory, how='left', on='timestamp')

            filepath = self.fault_dir / 'svc_metrics'
            os.makedirs(filepath, exist_ok=True)
            filepath = filepath / f'{svc}.csv'

            if os.path.isfile(filepath):
                original_df = pd.read_csv(filepath)
                df = pd.concat([original_df, df], ignore_index=True)

            df.set_index('timestamp')
            df.to_csv(filepath, index=False)

    def ctn_network(self, pod_name):
        """Query container network usage and return as a series"""
        ctn_network_query = self.get_ctn_network_promQL(pod_name)
        results = self.query_prometheus_range(ctn_network_query)
        return self.parse_ctn_metric(results)

    def ctn_memory(self, pod_name):
        """Query memory network usage and return as a series"""
        ctn_memory_query = self.get_ctn_memory_promQL(pod_name)
        results = self.query_prometheus_range(ctn_memory_query)
        return self.parse_ctn_metric(results)

    def parse_ctn_metric(self, results):
        """Parse container query results to pandas series"""
        values = results[0]['values']
        values = list(zip(*values))
        metric = pd.Series(values[1])
        return metric

    def node_network(self, instance):
        """Query node network usage and return a dataframe"""
        node_network_query = self.get_node_network_promQL(instance)
        results = self.query_prometheus_range(node_network_query)
        return self.parse_node_metric(results, 'node_network')

    def node_cpu(self, instance):
        """Query node cpu usage and return a dataframe"""
        node_cpu_query = self.get_node_cpu_promQL(instance)
        results = self.query_prometheus_range(node_cpu_query)
        return self.parse_node_metric(results, 'node_cpu')

    def node_memory(self, instance):
        """Query node cpu usage and return a dataframe"""
        node_memory_query = self.get_node_memory_promQL(instance)
        results = self.query_prometheus_range(node_memory_query)
        return self.parse_node_metric(results, 'node_memory')

    def parse_node_metric(self, results, key: str):
        """
        Parse container query results to pandas datafram

        must specify metrics key: node_network, node_cpu, or node_memory
        """
        values = results[0]['values']
        values = list(zip(*values))
        df = pd.DataFrame()
        df['timestamp'] = values[0]
        df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        df[key] = pd.Series(values[1])
        df[key] = df[key].astype('float64')
        return df

    def parser_mpg_results(self, DG, df, results):
        """Parse query results into DG provided and returns it"""
        for result in results:
            metric = result['metric']

            # for hosting edges
            if 'container' in metric:
                source = metric['container']
                destination = metric['instance']
                d = {'source':[source], 'destination': [destination]}
                new_df = pd.DataFrame(data=d)
                df = pd.concat([df, new_df], ignore_index=True)
                DG.add_edge(source, destination)
                
                DG._node[source]['type'] = 'service'
                DG._node[destination]['type'] = 'host'
                continue 

            source = metric['source_workload']
            destination = metric['destination_workload']
            d = {'source':[source], 'destination': [destination]}
            new_df = pd.DataFrame(data=d)
            df = pd.concat([df, new_df], ignore_index=True)
            DG.add_edge(source, destination)
            
            DG._node[source]['type'] = 'service'
            DG._node[destination]['type'] = 'service'

        return DG, df

    def mpg(self):
        """Query data, generate DAG, save the DAG in pickle"""
        DG = nx.DiGraph()
        df = pd.DataFrame(columns=['source', 'destination'])

        if self.tcp:
            tcp_edge_query = self.get_mpg_tcp_edges_promQL()
            results = self.query_prometheus(tcp_edge_query)
            DG, df = self.parser_mpg_results(DG, df, results)

        http_edge_query = self.get_mpg_http_edges_promQL()
        results = self.query_prometheus(http_edge_query)
        DG, df = self.parser_mpg_results(DG, df, results)

        hosting_edge_query = self.get_mpg_hosting_edges_promQL()
        results = self.query_prometheus(hosting_edge_query)
        DG, df = self.parser_mpg_results(DG, df, results)

        csv_filename = self.fault_dir / 'mpg.csv'
        df.to_csv(csv_filename, index=False)
        pickle_filename = self.fault_dir / 'mpg.pkl'
        save_as_pickle(pickle_filename, DG)

    def get_latency_promQL(self, reporter: str):
        """Simply return PromQL with reporter enbeded"""
        return 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter="%(reporter)s", destination_workload_namespace="%(namespace)s", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload, le))' % {'namespace': self.namespace, 'reporter': reporter} 
        
    def get_tcp_sent_promQL(self, reporter: str):
        """Simply return PromQL with reporter enbeded"""
        return  'sum(irate(istio_tcp_sent_bytes_total{reporter="%(reporter)s", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload) / 1000' % {'reporter': reporter}

    def get_ctn_cpu_promQL(self):
        """Simply return PromQL for container cpu usage for all containers"""
        return 'sum(rate(container_cpu_usage_seconds_total{namespace="%(namespace)s", container!~"POD|istio-proxy|"}[1m])) by (pod, instance, container)' % {'namespace': self.namespace}


    def get_ctn_network_promQL(self, pod_name: str):
        """Simply return PromQL for container network usage for a container"""
        return 'sum(rate(container_network_transmit_packets_total{namespace="%(namespace)s", pod="%(pod_name)s"}[1m])) / 1000 * sum(rate(container_network_transmit_packets_total{namespace="%(namespace)s", pod="%(pod_name)s"}[1m])) / 1000' % {'pod_name': pod_name, 'namespace':self.namespace}

    def get_ctn_memory_promQL(self, pod_name: str):
        """Simply return PromQL for container memory usage for a container"""
        return 'sum(rate(container_memory_working_set_bytes{namespace="%(namespace)s", pod="%(pod_name)s"}[1m])) / 1000' % {'pod_name': pod_name, 'namespace': self.namespace}

    def get_node_network_promQL(self, instance: str, device='ens3'):
        """Simply return PromQL for network usage for the a node"""
        return 'rate(node_network_transmit_packets_total{device="%(device)s", instance="%(instance)s"}[1m]) / 1000' % {'instance': instance, 'device': device}

    def get_node_cpu_promQL(self, instance: str):
        """Simply return PromQL for cpu usage for the a node"""
        return 'sum(rate(node_cpu_seconds_total{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="%(instance)s" }[1m])) / count(node_cpu_seconds_total{mode="system", instance="%(instance)s"})' % {'instance': instance},

    def get_node_memory_promQL(self, instance: str):
        """Simply return PromQL for memory usage for the a node"""
        return '1 - sum(node_memory_MemAvailable_bytes{instance="%(instance)s"}) / sum(node_memory_MemTotal_bytes{instance="%(instance)s"})' % {'instance': instance},

    def get_mpg_tcp_edges_promQL(self):
        """Simply return PromQL for mapping"""
        return 'sum(istio_tcp_received_bytes_total{destination_workload_namespace="%(namespace)s",destination_workload!="unknown", source_workload!="unknown"}) by (source_workload, destination_workload)' % {'namespace': self.namespace}

    def get_mpg_http_edges_promQL(self):
        """Simply return PromQL for mapping"""
        return 'sum(istio_requests_total{destination_workload_namespace="%(namespace)s", destination_workload!="unknown", source_workload!="unknown"}) by (source_workload, destination_workload)' % {'namespace': self.namespace}

    def get_mpg_hosting_edges_promQL(self):
        """Simply return PromQL for mapping"""
        return 'sum(container_cpu_usage_seconds_total{namespace="%(namespace)s", container!~"POD|istio-proxy|", container!="rabbitmq-exporter"}) by (instance, container)' % {'namespace': self.namespace}
