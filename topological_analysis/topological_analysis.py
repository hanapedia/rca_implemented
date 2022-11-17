#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import numpy as np
# import numpy as np

class Topology():
    def __init__(self, name='Topological analysis', loc_err=False):
        self.topology = nx.DiGraph()
        self.name = name
        self.loc_err = loc_err
        self.edge_color = 'gray'
        self.node_color = '#1f78b4'

    def load_csv(self, csv_path):
        self.df = pd.read_csv(csv_path, index_col=0)

    def set_topology(self, topo):
        self.topology = topo

    def set_edge_colors(self):
        edge_color = []
        anomalies_dict = nx.get_edge_attributes(self.topology, 'anomalous')
        for edge in self.topology.edges():
            if edge in anomalies_dict:
                edge_color.append('red')
            else:
                edge_color.append('gray')
        self.edge_color = edge_color

    # set edge color based on num invo weight
    def set_edge_colors_n(self):
        edge_color = []
        invos_dict = nx.get_edge_attributes(self.topology, 'num_invo')
        for edge in self.topology.edges():
            if edge in anomalies_dict:
                edge_color.append('red')
            else:
                edge_color.append('gray')
        self.edge_color = edge_color

    def set_node_colors_and_label(self):
        node_color = []
        node_label_dict = {}
        for node in self.topology.nodes(data=True):
            node_color.append(node[1]['color'])
            node_label_dict[node[0]] = node[1]['label']

        self.node_color = node_color
        self.node_label = node_label_dict

    def generate_topology(self, df: pd.DataFrame, row_labels: set, exclude=False, exclude_label='node', loc_err_conf={}):
        self.df = df
        self.exclude = exclude
        self.exclude_label = exclude_label
        self.row_labels = row_labels
        edge_attr_dict = defaultdict(dict)
        node_attr_dict = defaultdict(dict)

        # Check if predict column exists on the dataframe
        if self.loc_err and 'predict' not in self.df.columns:
            self.loc_err = False

        if self.loc_err and not loc_err_conf:
            raise Exception("Must provide loc_err_conf for Topology instance with loc_err enabled")
            
        for _, row in self.df.iterrows():
            # filter out some nodes
            if not self.exclude:
                if (self.exclude_label in row[self.row_labels[0]] or self.exclude_label in row[self.row_labels[1]]):
                    continue
            # generate DAG
            self.topology.add_edge(row[self.row_labels[0]], row[self.row_labels[1]])
            # edge attribute: number of invocation for the edge
            if 'num_invo' not in edge_attr_dict[(row[self.row_labels[0]], row[self.row_labels[1]])]:
                edge_attr_dict[(row[self.row_labels[0]], row[self.row_labels[1]])]['num_invo'] = 0
            edge_attr_dict[(row[self.row_labels[0]], row[self.row_labels[1]])]['num_invo'] += 1
            if self.loc_err and row['predict']:
                # edge attribute: boolean to indicate if the invocation was detected as anomalous
                edge_attr_dict[(row[self.row_labels[0]], row[self.row_labels[1]])]['anomalous'] = True


        if 'selected_features' in loc_err_conf:
            for k, v in loc_err_conf['selected_features'].items():
                edge_attr_dict = dict(edge_attr_dict)
                # edge attribute: features that were selected for anomaly detection of the edge
                edge_attr_dict[k]['selected_features'] = v
        nx.set_edge_attributes(self.topology, edge_attr_dict)

        # Set node fact
        if 'root_cause' in loc_err_conf and 'predictions' in loc_err_conf:
            predict_color = ['#a85832', '#a88e32', '#8ca832', '#50a832', '#32a871']
            regular_node_color = '#1f78b4'
            color_dict = {k: v for k, v in zip(loc_err_conf['predictions'], predict_color)}
            for node in self.topology.nodes():
                if node in color_dict:
                    node_attr_dict[node]['color'] = color_dict[node] 
                else:
                    node_attr_dict[node]['color'] = regular_node_color

                if node in loc_err_conf['root_cause']:
                    node_attr_dict[node]['label'] = f'{node}*(RC)*'
                else:
                    node_attr_dict[node]['label'] = node
        nx.set_node_attributes(self.topology, node_attr_dict)

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
            anomalies_in = 0
            for in_e in in_edges:
                invo_in.append(in_e[2]['num_invo'] )
                if 'anomalous' in in_e[2]:
                    anomalies_in += 1

            invo_out = [] 
            anomalies_out = 0
            for out_e in out_edges:
                invo_out.append(out_e[2]['num_invo']) 
                if 'anomalous' in out_e[2]:
                    anomalies_out += 1

            in_avg, in_var = calc_avg_and_var(invo_in)
            out_avg, out_var = calc_avg_and_var(invo_out)
            return_dict = {
                'in_edges': in_edges,
                'out_edges': out_edges,
                'num_in': num_in,
                'num_out': num_out,
                'num_out-in': num_out - num_in,
                'num_invo-in': sum(invo_in),
                'num_invo-out': sum(invo_out),
                'num_invo-in-avg': in_avg,
                'num_invo-out-avg': out_avg,
                'num_invo-in-var': in_var,
                'num_invo-out-var': out_var,
            } 

            if self.loc_err:
                return_dict['num_anomalous_in'] = anomalies_in
                return_dict['num_anomalous_out'] = anomalies_out

            return return_dict

        def calc_avg_and_var(invo_list):
            if len(invo_list) > 1:
                return np.mean(np.asarray(invo_list)), np.var(np.asarray(invo_list))
            return 'n/a', 'n/a'

            
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

    # TODO 
    # node and edge coloring using cm
    # default configs
    def draw(self, show, path='', edge_label=True, plot_opt={}):
        plot_opt_default = {}
        if 'ax' not in plot_opt:
            def_fig = plt.figure(figsize=[16,16],dpi=80)
            def_ax = def_fig.add_axes([0,0,1,1])
            def_ax.set_title(self.name)
            plot_opt_default['ax'] = def_ax

        plot_opt = plot_opt_default | plot_opt
        if self.loc_err:
            self.set_edge_colors()
            self.set_node_colors_and_label()
            el = nx.get_edge_attributes(self.topology, 'selected_features')
            bbox = dict(boxstyle='round', ec=(0.0, 1.0, 1.0, 0), fc=(0.0, 1.0, 1.0, 0))
            nx.draw_networkx_edge_labels(self.topology, pos=nx.nx_pydot.graphviz_layout(self.topology, prog='dot'), ax=plot_opt['ax'], edge_labels = el, font_size=8,verticalalignment='center_baseline', label_pos= 0.5, rotate=True, bbox=bbox)

        nx.draw_networkx(self.topology, pos=nx.nx_pydot.graphviz_layout(self.topology, prog='dot'), ax=plot_opt['ax'], node_size=300, font_size=11, width=1, arrowsize=10, edge_color=self.edge_color, node_color=self.node_color, labels=self.node_label)
        if edge_label:
            el = nx.get_edge_attributes(self.topology, 'num_invo')
            bbox = dict(boxstyle='round', ec=(0.0, 1.0, 1.0, 0), fc=(0.0, 1.0, 1.0, 0))
            nx.draw_networkx_edge_labels(self.topology, pos=nx.nx_pydot.graphviz_layout(self.topology, prog='dot'), ax=plot_opt['ax'], edge_labels = el, font_size=8, verticalalignment='bottom', label_pos= 0.5, rotate=True, bbox=bbox)

        if show:
            plt.show()
        if path:
            plt.savefig(path)
            plt.close()

