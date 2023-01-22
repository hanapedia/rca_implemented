import pandas as pd
import numpy as np
import networkx as nx
import csv
from pathlib import Path
from sklearn.cluster import Birch
from sklearn import preprocessing

from ._utils import load_pickle

class Microrca:
    """Implementation of MicorRCA"""
    def __init__(self, dataset_dir: str, results_dir: str, include_db=False) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir)
        self.include_db = include_db
        self.smoothing_window = 12
        self.ad_threshold = 0.05
        self.alpha = 0.55

    def run(self):
        latency_df_source = self.latency_source_50()
        latency_df_destination = self.latency_destination_50()
        latency_df = latency_df_destination.iloc[:, 1:].add(latency_df_source.iloc[:, 1:], fill_value=0)
        
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

        filename = self.results_dir / 'results.csv'
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(anomalies)#.insert(0, 'anomalous services'))
            writer.writerow(anomaly_score)#.insert(0, 'anomaly score raw'))
            writer.writerow(anomaly_score_new)#.insert(0, 'anomaly score no db'))


    def birch_ad_with_smoothing(self, latency_df):
        """
        anomaly detection on response time of service invocation. 
        input: response times of service invocations, threshold for birch clustering
        output: anomalous service invocation
        """
        anomalies = []
        if self.include_db:
            for svc, latency in latency_df.items():
                # No anomaly detection in db
                # if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
                if svc != 'timestamp' and 'Unnamed' not in svc: # and 'rabbitmq' not in svc and 'db' not in svc:
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
        else:
            for svc, latency in latency_df.items():
                # No anomaly detection in db
                # if svc != 'timestamp' and 'Unnamed' not in svc: # and 'rabbitmq' not in svc and 'db' not in svc:
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

    def anomaly_subgraph(self, DG, anomalies, latency_df):
        """
        Get the anomalous subgraph and rank the anomalous services
        input: 
          DG: attributed graph
          anomlies: anoamlous service invocations
          latency_df: service invocations from data collection
          latency_dff: aggregated service invocation
        output:
          anomalous scores 
        
        Get reported anomalous nodes
        """
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

    def node_weight(self, svc, anomaly_graph, baseline_df):
        """Add weight to the edges of anomaly graph""" 
        #Get the average weight of the in_edges
        in_edges_weight_avg = 0.0
        num = 0
        for _, _, data in anomaly_graph.in_edges(svc, data=True):
            num = num + 1
            in_edges_weight_avg = in_edges_weight_avg + data['weight']
        if num > 0:
            in_edges_weight_avg  = in_edges_weight_avg / num

        filename = self.dataset_dir / 'svc_metrics' / f'{svc}.csv' 
        df = pd.read_csv(filename)
        # node_cols = ['node_cpu', 'node_network', 'node_memory']
        node_cols = ['node_cpu', 'node_memory']
        max_corr = 0.01
        metric = 'node_cpu'
        for col in node_cols:
            temp = abs(baseline_df[svc].corr(df[col]))
            if temp > max_corr:
                max_corr = temp
                metric = col
        data = in_edges_weight_avg * max_corr
        return data, metric

    def svc_personalization(self, svc, anomaly_graph, baseline_df):
        """Calculate the personalization vector""" 
        filename = self.dataset_dir / 'svc_metrics' / f'{svc}.csv' 
        df = pd.read_csv(filename)
        # ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
        ctn_cols = ['ctn_cpu', 'ctn_memory']
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

    def latency_source_50(self):
        """Load latency_source_50 df from csv"""
        latency_source_path = self.dataset_dir / 'latency_source_50.csv'
        df = pd.read_csv(latency_source_path)
        return df

    def latency_destination_50(self):
        """Load latency_destination_50 df from csv"""
        latency_destination_path = self.dataset_dir / 'latency_destination_50.csv'
        df = pd.read_csv(latency_destination_path)
        return df

    def mpg(self):
        """Load DAG from pickle"""
        dag_path = self.dataset_dir / 'mpg.pkl'
        DG = load_pickle(dag_path)
        return DG
