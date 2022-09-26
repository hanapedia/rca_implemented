#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import networkx as nx
import csv
# import argparse
#import matplotlib.pyplot as plt

from sklearn.cluster import Birch
from sklearn import preprocessing

## =========== Data collection ===========
class Microrca:
    """Implements MicroRCA. 

    Parameters
    ---------
    node_dict: dictionary
        Dictionary containing the node name as a key and its ip address with prometheus node exporter port. 
        Egs.  
        node_dict = {
                        'cp1' : '192.168.100.254:9100',
                        'cp2' : '192.168.100.87:9100',
                        'node1' : '192.168.100.36:9100',
                        'node2' : '192.168.100.210:9100',
                        'node3' : '192.168.100.43:9100',
                }

    folder_path: string
        Folder path for saving the query and analysis results

    len_second: integer 
        number in seconds for duration to be analyzed

    prom_url: string
        Prometheus API url

    """
    # Constructor
    def __init__(self, node_dict, folder_path, len_second, prom_url, k_namespace='sock-shop', metric_step = '14s', smoothing_window = 12, alpha = 0.55, ad_threshold = 0.05):
        self.node_dict = node_dict
        self.folder_path = folder_path
        self.prom_url = prom_url 
        self.prom_url_range = prom_url + '_range' 
        self.len_second = len_second 
        self.k_namespace = k_namespace
        self.metric_step = metric_step
        self.smoothing_window = smoothing_window
        self.alpha = alpha
        self.ad_threshold = ad_threshold

    # Run microrca
    def run(self, faults_name, end_time, results_csv=''):
        self.faults_name = faults_name
        self.end_time = end_time
        self.start_time = end_time - self.len_second
        latency_df_source = self.latency_source_50()
        latency_df_destination = self.latency_destination_50()
        latency_df = latency_df_destination.iloc[:, 1:].add(latency_df_source.iloc[:, 1:], fill_value=0)
        
        self.svc_metrics()
        
        DG = self.mpg()

        # anomaly detection on response time of service invocation
        anomalies = self.birch_ad_with_smoothing(latency_df)
        print(anomalies)
        
        # get the anomalous service
        anomaly_nodes = []
        for anomaly in anomalies:
            edge = anomaly.split('_')
            anomaly_nodes.append(edge[1])
        
        anomaly_nodes = set(anomaly_nodes)
         
        anomaly_score = self.anomaly_subgraph(DG, anomalies, latency_df)
        print(anomaly_score)

        # Remove entries with db
        anomaly_score_new = []
        for anomaly_target in anomaly_score:
            node = anomaly_target[0]
            if DG._node[node]['type'] == 'service':
                anomaly_score_new.append(anomaly_target)
        print(anomaly_score_new)

        filename = self.faults_name + '/0_results.csv'
        with open(self.folder_path + filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(anomalies)#.insert(0, 'anomalous services'))
            writer.writerow(anomaly_score)#.insert(0, 'anomaly score raw'))
            writer.writerow(anomaly_score_new)#.insert(0, 'anomaly score no db'))

        labeled_score = [self.faults_name, *anomaly_score_new]
        print(labeled_score)
        if results_csv != '':
            with open(results_csv, 'a') as f:
                writer = csv.writer(f)
                # writer.writerow(anomalies)#.insert(0, 'anomalous services'))
                # writer.writerow(anomaly_score)#.insert(0, 'anomaly score raw'))
                writer.writerow(labeled_score)#.insert(0, 'anomaly score no db'))
                writer.writerow([])

    # Retrieve latency with reporter = source
    def latency_source_50(self):
        latency_df = pd.DataFrame()
        response = requests.get(self.prom_url_range,
                                params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter="source", destination_workload_namespace="%(namespace)s", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload, le))' % {'namespace': self.k_namespace},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']

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
            latency_df[name] = latency_df[name].astype('float64')  * 1000

        response = requests.get(self.prom_url_range,
                                params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter="source", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload) / 1000',
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']

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
            latency_df[name] = latency_df[name].astype('float64').rolling(window=self.smoothing_window, min_periods=1).mean()

        filename = self.faults_name + '/latency_source_50.csv'
        latency_df.set_index('timestamp')
        latency_df.to_csv(self.folder_path + filename)
        return latency_df
    
    # Retrieve latency with reporter = destination
    def latency_destination_50(self):
        latency_df = pd.DataFrame()
        response = requests.get(self.prom_url_range,
                                params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="%(namespace)s", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload, le))' % {'namespace': self.k_namespace},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']

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
            latency_df[name] = latency_df[name].astype('float64')  * 1000

        response = requests.get(self.prom_url_range,
                                params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter="destination", destination_workload!="unknown", source_workload!="unknown"}[1m])) by (destination_workload, source_workload) / 1000',
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']

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
            latency_df[name] = latency_df[name].astype('float64').rolling(window=self.smoothing_window, min_periods=1).mean()

        filename = self.faults_name + '/latency_destination_50.csv'
        latency_df.set_index('timestamp')
        latency_df.to_csv(self.folder_path + filename)
        return latency_df

    # Retrieve service metrics
    def svc_metrics(self):
        response = requests.get(self.prom_url_range,
                                params={'query': 'sum(rate(container_cpu_usage_seconds_total{namespace="%(namespace)s", container!~"POD|istio-proxy|", container!="rabbitmq-exporter"}[1m])) by (pod, instance, container)' % {'namespace': self.k_namespace},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']

        for result in results:
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

            df['ctn_network'] = self.ctn_network(pod_name)
            df['ctn_network'] = df['ctn_network'].astype('float64')
            df['ctn_memory'] = self.ctn_memory(pod_name)
            df['ctn_memory'] = df['ctn_memory'].astype('float64')

            instance = self.node_dict[nodename]

            df_node_cpu = self.node_cpu(instance)
            df = pd.merge(df, df_node_cpu, how='left', on='timestamp')


            df_node_network = self.node_network(instance)
            df = pd.merge(df, df_node_network, how='left', on='timestamp')

            df_node_memory = self.node_memory(instance)
            df = pd.merge(df, df_node_memory, how='left', on='timestamp')
        

            filename = self.faults_name + '/' + svc + '.csv'
            df.set_index('timestamp')
            df.to_csv(self.folder_path + filename)

    # Retrieve container network metrics
    def ctn_network(self, pod_name):
        response = requests.get(self.prom_url_range,
                                params={'query': 'sum(rate(container_network_transmit_packets_total{namespace="%(namespace)s", pod="%(pod_name)s"}[1m])) / 1000 * sum(rate(container_network_transmit_packets_total{namespace="%(namespace)s", pod="%(pod_name)s"}[1m])) / 1000' % {'pod_name': pod_name, 'namespace': self.k_namespace},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']

        values = results[0]['values']

        values = list(zip(*values))
        metric = pd.Series(values[1])
        return metric

    # Retrieve container memory metrics
    def ctn_memory(self, pod_name):
        response = requests.get(self.prom_url_range,
                                params={'query': 'sum(rate(container_memory_working_set_bytes{namespace="%(namespace)s", pod="%(pod_name)s"}[1m])) / 1000' % {'pod_name': pod_name, 'namespace': self.k_namespace},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']

        values = results[0]['values']

        values = list(zip(*values))
        metric = pd.Series(values[1])
        return metric

    # Retrieve node network metrics
    def node_network(self, instance):
        response = requests.get(self.prom_url_range,
                                params={'query': 'rate(node_network_transmit_packets_total{device="ens3", instance="%(instance)s"}[1m]) / 1000' % {'instance': instance},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']
        values = results[0]['values']

        values = list(zip(*values))
        df = pd.DataFrame()
        df['timestamp'] = values[0]
        df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        df['node_network'] = pd.Series(values[1])
        df['node_network'] = df['node_network'].astype('float64')
        return df

    # Retrieve node cpu metrics
    def node_cpu(self, instance):
        response = requests.get(self.prom_url_range,
                                params={'query': 'sum(rate(node_cpu_seconds_total{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="%(instance)s" }[1m])) / count(node_cpu_seconds_total{mode="system", instance="%(instance)s"})' % {'instance': instance},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']
        values = results[0]['values']
        values = list(zip(*values))
        df = pd.DataFrame()
        df['timestamp'] = values[0]
        df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        df['node_cpu'] = pd.Series(values[1])
        df['node_cpu'] = df['node_cpu'].astype('float64')
        return df

    # Retrieve node memory metrics
    def node_memory(self, instance):
        response = requests.get(self.prom_url_range,
                                params={'query': '1 - sum(node_memory_MemAvailable_bytes{instance="%(instance)s"}) / sum(node_memory_MemTotal_bytes{instance="%(instance)s"})' % {'instance': instance},
                                        'start': self.start_time,
                                        'end': self.end_time,
                                        'step': self.metric_step})
        results = response.json()['data']['result']
        values = results[0]['values']

        values = list(zip(*values))
        df = pd.DataFrame()
        df['timestamp'] = values[0]
        df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        df['node_memory'] = pd.Series(values[1])
        df['node_memory'] = df['node_memory'].astype('float64')
        return df

    # Create Graph
    def mpg(self):
        DG = nx.DiGraph()
        df = pd.DataFrame(columns=['source', 'destination'])
        response = requests.get(self.prom_url,
                                params={'query': 'sum(istio_tcp_received_bytes_total{destination_workload!="unknown", source_workload!="unknown"}) by (source_workload, destination_workload)'})
        results = response.json()['data']['result']

        for result in results:
            metric = result['metric']
            source = metric['source_workload']
            destination = metric['destination_workload']
            d = {'source':[source], 'destination': [destination]}
            new_df = pd.DataFrame(data=d)
            df = pd.concat([df, new_df], ignore_index=True)
            DG.add_edge(source, destination)
            
            DG._node[source]['type'] = 'service'
            DG._node[destination]['type'] = 'service'

        response = requests.get(self.prom_url,
                                params={'query': 'sum(istio_requests_total{destination_workload_namespace="%(namespace)s", destination_workload!="unknown", source_workload!="unknown"}) by (source_workload, destination_workload)' % {'namespace': self.k_namespace}})
        results = response.json()['data']['result']

        for result in results:
            metric = result['metric']
            
            source = metric['source_workload']
            destination = metric['destination_workload']
            d = {'source':[source], 'destination': [destination]}
            new_df = pd.DataFrame(data=d)
            df = pd.concat([df, new_df], ignore_index=True)
            DG.add_edge(source, destination)
            
            DG._node[source]['type'] = 'service'
            DG._node[destination]['type'] = 'service'

        response = requests.get(self.prom_url,
                                params={'query': 'sum(container_cpu_usage_seconds_total{namespace="%(namespace)s", container!~"POD|istio-proxy|", container!="rabbitmq-exporter"}) by (instance, container)' % {'namespace': self.k_namespace}})
        results = response.json()['data']['result']
        for result in results:
            metric = result['metric']
            if 'container' in metric:
                source = metric['container']
                destination = metric['instance']
                d = {'source':[source], 'destination': [destination]}
                new_df = pd.DataFrame(data=d)
                df = pd.concat([df, new_df], ignore_index=True)
                DG.add_edge(source, destination)
                
                DG._node[source]['type'] = 'service'
                DG._node[destination]['type'] = 'host'

        filename = self.faults_name + '/mpg.csv'
        df.to_csv(self.folder_path + filename)
        return DG


    # Anomaly Detection
    def birch_ad_with_smoothing(self, latency_df):
        # anomaly detection on response time of service invocation. 
        # input: response times of service invocations, threshold for birch clustering
        # output: anomalous service invocation
        
        anomalies = []
        for svc, latency in latency_df.iteritems():
            # No anomaly detection in db
            if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
                latency = latency.rolling(window=self.smoothing_window, min_periods=1).mean()
                x = np.array(latency)
                x = np.where(np.isnan(x), 0, x)
                normalized_x = preprocessing.normalize([x])

                X = normalized_x.reshape(-1,1)

                brc = Birch(branching_factor=50, n_clusters=None, threshold=self.ad_threshold, compute_labels=True)
                brc.fit(X)
                brc.predict(X)

                labels = brc.labels_
                n_clusters = np.unique(labels).size
                if n_clusters > 1:
                    anomalies.append(svc)
        return anomalies

    # Add weight to the edges of anomaly graph 
    def node_weight(self, svc, anomaly_graph, baseline_df):
        #Get the average weight of the in_edges
        in_edges_weight_avg = 0.0
        num = 0
        for _, _, data in anomaly_graph.in_edges(svc, data=True):
            num = num + 1
            in_edges_weight_avg = in_edges_weight_avg + data['weight']
        if num > 0:
            in_edges_weight_avg  = in_edges_weight_avg / num

        filename = self.faults_name + '/' + svc + '.csv'
        df = pd.read_csv(self.folder_path + filename)
        node_cols = ['node_cpu', 'node_network', 'node_memory']
        max_corr = 0.01
        metric = 'node_cpu'
        for col in node_cols:
            temp = abs(baseline_df[svc].corr(df[col]))
            if temp > max_corr:
                max_corr = temp
                metric = col
        data = in_edges_weight_avg * max_corr
        return data, metric

    # Calculate the personalization vector
    def svc_personalization(self, svc, anomaly_graph, baseline_df):
        filename = self.faults_name + '/' + svc + '.csv'
        df = pd.read_csv(self.folder_path + filename)
        ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
        max_corr = 0.01
        metric = 'ctn_cpu'
        for col in ctn_cols:
            temp = abs(baseline_df[svc].corr(df[col]))     
            if temp > max_corr:
                max_corr = temp
                metric = col


        edges_weight_avg = 0.0
        num = 0
        for _, v, data in anomaly_graph.in_edges(svc, data=True):
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

        for _, v, data in anomaly_graph.out_edges(svc, data=True):
            if anomaly_graph._node[v]['type'] == 'service':
                num = num + 1
                edges_weight_avg = edges_weight_avg + data['weight']

        edges_weight_avg  = edges_weight_avg / num

        personalization = edges_weight_avg * max_corr

        return personalization, metric

    # Construct anomaly subgraph
    def anomaly_subgraph(self, DG, anomalies, latency_df):
        # Get the anomalous subgraph and rank the anomalous services
        # input: 
        #   DG: attributed graph
        #   anomlies: anoamlous service invocations
        #   latency_df: service invocations from data collection
        #   latency_dff: aggregated service invocation
        # output:
        #   anomalous scores 
        
        # Get reported anomalous nodes
        edges = []
        nodes = []
        baseline_df = pd.DataFrame()
        edge_df = {}
        for anomaly in anomalies:
            edge = anomaly.split('_')
            edges.append(tuple(edge))
            svc = edge[1]
            nodes.append(svc)
            baseline_df[svc] = latency_df[anomaly]
            edge_df[svc] = anomaly

        nodes = set(nodes)

        personalization = {}
        for node in DG.nodes():
            if node in nodes:
                personalization[node] = 0

        # Get the subgraph of anomaly
        anomaly_graph = nx.DiGraph()
        for node in nodes:
            for u, v, data in DG.in_edges(node, data=True):
                edge = (u,v)
                if edge in edges:
                    data = self.alpha
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[v].corr(latency_df[normal_edge])

                data = round(data, 3)
                anomaly_graph.add_edge(u,v, weight=data)
                anomaly_graph._node[u]['type'] = DG.nodes[u]['type']
                anomaly_graph._node[v]['type'] = DG.nodes[v]['type']

           # Set personalization with container resource usage
            for u, v, data in DG.out_edges(node, data=True):
                edge = (u,v)
                if edge in edges:
                    data = self.alpha
                else:

                    if DG._node[v]['type'] == 'host':
                        data, _ = self.node_weight(u, anomaly_graph, baseline_df)
                    else:
                        normal_edge = u + '_' + v
                        data = baseline_df[u].corr(latency_df[normal_edge])
                data = round(data, 3)
                anomaly_graph.add_edge(u,v, weight=data)
                anomaly_graph._node[u]['type'] = DG.nodes[u]['type']
                anomaly_graph._node[v]['type'] = DG.nodes[v]['type']

        for node in nodes:
            max_corr, _ = self.svc_personalization(node, anomaly_graph, baseline_df)
            personalization[node] = max_corr / anomaly_graph.degree(node)

        anomaly_graph = anomaly_graph.reverse(copy=True)
        edges = list(anomaly_graph.edges(data=True))

        for u, v, d in edges:
            if anomaly_graph._node[node]['type'] == 'host':
                anomaly_graph.remove_edge(u,v)
                anomaly_graph.add_edge(v,u,weight=d['weight'])

        anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000, tol=0.1)

        anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)
        return anomaly_score

