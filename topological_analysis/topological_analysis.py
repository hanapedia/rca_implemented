#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt

class Topology():
    def __init__(self, rca_path='', csv_path='', hosting_node=False, hosting_node_label='node'):
        self.rca_path = rca_path
        self.csv_path = csv_path
        self.hosting_node = hosting_node
        self.hosting_node_label = hosting_node_label
        self.topology = nx.DiGraph()

        if rca_path == '' and csv_path == '':
            raise Exception("Either one of rca_path or csv_path must be defined")

        if csv_path and rca_path:
            self.rca_path = ''

        if rca_path and csv_path == '':
            self.csv_path = os.listdir(rca_path)[0]

        self.df = pd.read_csv(self.csv_path, index_col=0)

        self.generate_topology()

    def generate_topology(self):
        for _, row in self.df.iterrows():
            if not self.hosting_node and (self.hosting_node_label in row['source'] or self.hosting_node_label in row['destination']):
                continue
            self.topology.add_edge(row['source'], row['destination'])

    def pagerank(self):
        self.reversed_topology = self.topology.reverse(copy=True)
        return nx.pagerank(self.reversed_topology)

    def draw(self):
        nx.draw_networkx(self.topology)
        plt.show()
        

