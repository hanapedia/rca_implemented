import unittest
import time
import csv
import pandas as pd
from pathlib import Path

from query.query import MicroRCAQuery

class QueryTests(unittest.TestCase):
    prom_url = "http://localhost:9090/api/v1/query"
    ns = "chc10s"
    fault_dir = Path("/Users/hirokihanada/code/src/github.com/hanapedia/chaos-experiments/experiment/generated/chain_fanout_single_db/chc10s/")

    len_seconds = 180
    end_time = time.time()
    metric_step = '5s'
    node_dict = {
        "experiment-cluster-cp1" : "192.168.100.130:9100",
        "experiment-cluster-cp2" : "192.168.100.9:9100",
        "experiment-cluster-node1" : "192.168.100.54:9100",
        "experiment-cluster-node2" : "192.168.100.154:9100",
        "experiment-cluster-node3" : "192.168.100.83:9100",
    }

    mq = MicroRCAQuery(
            prom_url=prom_url, 
            namespace=ns,
            fault_dir=fault_dir,
            len_seconds=len_seconds,
            end_time= end_time,
            metric_step=metric_step,
            node_dict=node_dict
            )

    def test_queries(self):
        self.mq.latency_50(reporter="source")
        self.mq.latency_50(reporter="destination")
        self.mq.svc_metrics()
        self.mq.mpg()

        self.assertTrue(True)

    def test_load_node_dict(self):
        """Load node mapping from csv and parse it into a dict"""
        node_dict_path = self.fault_dir / "node_dict.csv"
        node_dict = {}
        with open(node_dict_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_dict[row[0]] = row[1]

        print(node_dict)

    def test_csv_to_df(self):
        source = "/Users/hirokihanada/code/src/github.com/hanapedia/chaos-experiments/experiment/generated/chain_fanout-v3/chc5s/datasets/microrca/chain-1_delay_3/latency_source_50.csv"
        dest = "/Users/hirokihanada/code/src/github.com/hanapedia/chaos-experiments/experiment/generated/chain_fanout-v3/chc5s/datasets/microrca/chain-1_delay_3/latency_destination_50.csv"
        source = pd.read_csv(source)
        dest = pd.read_csv(dest)
        source = source.loc[:, ["timestamp", "gateway_chain-1", "chain-1_chain-2"]]
        dest = dest.loc[:, ["timestamp", "gateway_chain-1", "chain-1_chain-2"]]
        cct_ignore = pd.concat([source, dest], ignore_index=True)
        cct= pd.concat([source, dest], ignore_index=False)
        print(cct_ignore)
        print(cct)

