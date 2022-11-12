#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
# import numpy as np

class Topology():
    def __init__(self):
        self.topology = nx.DiGraph()

    def load_csv(self, csv_path):
        self.df = pd.read_csv(csv_path, index_col=0)

    def set_df(self, df):
        self.df = df

    def set_topology(self, topo):
        self.topology = topo

    def generate_topology(self, row_labels: set, exclude=False, exclude_label='node'):
        self.exclude = exclude
        self.exclude_label = exclude_label
        self.row_labels = row_labels
        attr_dict = defaultdict(lambda: defaultdict(int))
        for _, row in self.df.iterrows():
            if not self.exclude:
                if (self.exclude_label in row[self.row_labels[0]] or self.exclude_label in row[self.row_labels[1]]):
                    continue
            self.topology.add_edge(row[self.row_labels[0]], row[self.row_labels[1]])
            attr_dict[(row[self.row_labels[0]], row[self.row_labels[1]])]['num_invo'] += 1

        nx.set_edge_attributes(self.topology, attr_dict)

    def pagerank(self):
        self.reversed_topology = self.topology.reverse(copy=True)
        pr = nx.pagerank(self.reversed_topology) 
        self.pr = {k: v for k, v in sorted(pr.items(), key=lambda item: item[1], reverse=True)}
        return self.pr

    def get_io_egdes(self):
        def format_node(node):
            in_edges = list(self.topology.in_edges(node, data=True))
            out_edges = list(self.topology.out_edges(node, data=True))
            num_in = len(in_edges)
            num_out = len(out_edges)

            invo_in = []
            for in_e in in_edges:
                invo_in.append(in_e[2]['num_invo'] )

            invo_out = [] 
            for out_e in out_edges:
                invo_out.append(out_e[2]['num_invo']) 

            in_avg, in_var = calc_avg_and_var(invo_in)
            out_avg, out_var = calc_avg_and_var(invo_out)
            return {
                'in_edges': in_edges,
                'out_edges': out_edges,
                'num_in': num_in,
                'num_out': num_out,
                'num_out-in': num_out - num_in,
                'num_invo-in': sum(invo_in),
                'num_invo-in-avg': in_avg,
                'num_invo-in-var': in_var,
                'num_invo-out': sum(invo_out),
                'num_invo-out-avg': out_avg,
                'num_invo-out-var': out_var 
            } 

        def calc_avg_and_var(invo_list):
            if len(invo_list) > 1:
                return np.mean(np.asarray(invo_list)), np.var(np.asarray(invo_list))
            return 'n/a', 'n/a'

            
        self.invo_dict = nx.get_edge_attributes(self.topology, 'num_invo')
        self.nodes_desc = dict(map(lambda k:  (k, format_node(k)), self.topology.nodes()))


    def rank_nodes(self, order: str):
        self.get_io_egdes()
        if order == 'out':
            return {k: v for k, v in sorted(self.nodes_desc.items(), key=lambda x: x[1]['num_out'], reverse=True)}

        elif order == 'in':
            return {k: v for k, v in sorted(self.nodes_desc.items(), key=lambda x: x[1]['num_in'], reverse=True)}

        elif order == 'diff':
            return {k: v for k, v in sorted(self.nodes_desc.items(), key=lambda x: x[1]['num_out-in'], reverse=True)}

        elif order == 'invo-in':
            return {k: v for k, v in sorted(self.nodes_desc.items(), key=lambda x: x[1]['num_invo-in'], reverse=True)}

        elif order == 'invo-out':
            return {k: v for k, v in sorted(self.nodes_desc.items(), key=lambda x: x[1]['num_invo-out'], reverse=True)}

        else:
            raise Exception("order must be either 'in', 'out', or 'diff'")

    def draw(self, edge_label=True, svg='./fig.svg'):
        nx.draw_networkx(self.topology, pos=nx.nx_pydot.graphviz_layout(self.topology, prog='dot'), node_size=300, font_size=9,edge_color='gray')
        if edge_label:
            el = nx.get_edge_attributes(self.topology, 'num_invo')
            nx.draw_networkx_edge_labels(self.topology, pos=nx.nx_pydot.graphviz_layout(self.topology, prog='dot'), edge_labels = el, font_size=7, horizontalalignment='left', label_pos= 0.35, rotate=False)
        # nx.draw_networkx(self.topology)
        plt.show()

        # plt.figure(figsize=[6, 6])
        # plt.axis('off')
        # plt.gca().set_position([0, 0, 1, 1])
        # plt.savefig(svg)
        

