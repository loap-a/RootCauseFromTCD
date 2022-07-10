# -*- coding: UTF-8 -*-
from __future__ import print_function
from ast import Sub
from multiprocessing.spawn import spawn_main
from posixpath import split
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
import time
import datetime
import threading
import multiprocessing as mp
import itertools as it
from functools import cmp_to_key

# 完善了多进程+多线程数据加载
# 保留前一版本的caller, callee, caller_callee以方便查找(反正内存够用)
# 经过反复修改, 将span作为代码中最根本的数据单元
# call graph 中的边

class Span:
    # caller, callee 类型为Node
    def __init__(self, caller, callee, metric, entry_span=None, is_entry=False):
        self.caller = caller
        self.callee = callee
        self.metric = metric
        self.entry_span = entry_span

        self.discovered = False
        self.anomaly_detected = False

        self.callgraph_index = -1

        self.qpm_anomaly = False
        self.ec_anomaly = False
        self.rt_anomaly = False

        self.qpm_beta = 0.0
        self.ec_beta = 0.0
        self.rt_beta = 0.0

        self.is_entry = is_entry

    def __eq__(self, span):
        if span == None:
            return False
        return self.caller == span.caller and self.callee == span.callee

    def __hash__(self):
        return hash(self.caller)+hash(self.callee)

    def __str__(self):
        return str(self.caller)+' \n'+str(self.callee)

    def get_caller(self):
        return self.caller

    def get_callee(self):
        return self.callee

    def normalize_metric(self):
        request_list = split_data(self.metric[0])
        duration_list = split_data(self.metric[1])
        exception_list = split_data(self.metric[2])
        timeout_list = split_data(self.metric[3])

        self.qpm = request_list
        self.ec = [np.nan]*1440
        self.rt = [np.nan]*1440

        for exception, timeout, index in zip(exception_list, timeout_list, range(0,1440)):
            if (not np.isnan(exception)) and (not np.isnan(timeout)):
                self.ec[index] = exception + timeout
        
        for duration, request, index in zip(duration_list, request_list, range(0, 1440)):
            if (not np.isnan(duration)) and (not np.isnan(duration)):
                self.rt[index] = duration/request

    def compute_pearsonr_to_entry_span(self):

        if self.is_entry == True:
            self.entry_span = self

        compare_qpm = self.entry_span.qpm
        compare_ec = self.entry_span.ec
        compare_rt = self.entry_span.rt

        data1_qpm = []
        data2_qpm = []

        data1_ec = []
        data2_ec = []

        data1_rt = []
        data2_rt = []

        for data_point, compare_point in zip(self.qpm, compare_qpm):
            if (not np.isnan(data_point)) and (not np.isnan(compare_point)):
                data1_qpm.append(data_point)
                data2_qpm.append(compare_point)
        
        for data_point, compare_point in zip(self.ec, compare_ec):
            if (not np.isnan(data_point)) and (not np.isnan(compare_point)):
                data1_ec.append(data_point)
                data2_ec.append(compare_point)

        for data_point, compare_point in zip(self.rt, compare_rt):
            if (not np.isnan(data_point)) and (not np.isnan(compare_point)):
                data1_rt.append(data_point)
                data2_rt.append(compare_point)


        if len(data1_qpm) > 20:
            self.qpm_similarity = abs(pearsonr(data1_qpm, data2_qpm)[0])
        if len(data1_ec) > 20:
            self.ec_similarity = abs(pearsonr(data1_ec, data2_ec)[0])
        if len(data1_rt) > 20:
            self.rt_similarity = abs(pearsonr(data1_rt, data2_rt)[0])



# caller or callee
class Node:
    # type代表Node的类型, 为表直观, 其值为'caller' or 'callee'
    def __init__(self, server, service, method, set):
        self.server = server
        self.service = service
        self.method = method
        self.set = set

        self.depth = 0
        self.list_index = -1
        self.callgraph_index = -1

    def __eq__(self, node):
        if node == None:
            return False
        return self.server == node.server and self.service == node.service and self.method == node.method and self.set == node.set

    def __hash__(self):
        return hash(self.server+self.service+self.method+self.set)

    def __str__(self):
        return '('+self.server+','+self.service+','+self.method+','+self.set+')'

    def __add__(self, node):
        return self.get_turple() + node.get_turple()

    def get_turple(self):
        return (self.server, self.service, self.method, self.set)

class Root_Cause:
    def __init__(self, turple, root_score):
        self.turple = turple
        self.root_score = root_score
    
    def __str__(self):
        return str(self.turple)+', root score: '+str(self.root_score)

    def __eq__(self, root_cause):
        if root_cause == None:
            return False
        else:
            return self.turple == root_cause.turple

class Pearsonr_Pruning:

    def __init__(self, downstream_span, compare_span_list):
        self.downstream_span = downstream_span
        self.compare_span_list = compare_span_list
        self.lack_data=False
        self.compute_pearsonr()

    def compute_pearsonr(self):
        self.compare_qpm = [np.nan]*1440
        for index in range(0, 1440):
            flag = True
            for span in self.compare_span_list:
                if np.isnan(span.qpm[index]):
                    flag = False
            if flag:
                self.compare_qpm[index] = 0.0
                for span in self.compare_span_list:
                    self.compare_qpm[index] = self.compare_qpm[index] + span.qpm[index]
        data1 = []
        data2 = []
        for downstream_qpm_point, compare_qpm_point in zip(self.downstream_span.qpm, self.compare_qpm):
            if (not np.isnan(downstream_qpm_point)) and (not np.isnan(compare_qpm_point)):
                data1.append(downstream_qpm_point)
                data2.append(compare_qpm_point)

        if len(data1)<20:
            self.similarity = 0
            self.p = 1
            self.lack_data = True
    
        else:
            self.similarity, self.p = pearsonr(data1, data2)
            self.similarity = abs(self.similarity)

    def __str__(self):
        return str(self.similarity)


class Span_Chain:

    def __init__(self):
        self.span_list = []

    def add_span_to_top(self, span):
        self.span_list.insert(0, span)

    def add_span_to_bottom(self, span):
        self.span_list.append(span)

class SubSubGraph:
    def __init__(self, span_list):
        self.span_list = span_list
        self.node_list = []
        self.adjancy_matrix = []

        self.end_nodes = []
        self.end_spans = []
        self.enter_spans = []
        self.enter_nodes = []
        
        self.span_chains = []

        for span in self.span_list:
            self.node_list.append(span.get_caller())
            self.node_list.append(span.get_callee())
        self.node_list = (list(set(self.node_list)))

        nodes_number = len(self.node_list)
        for i in range(0, nodes_number):
            temp_list = [None]*nodes_number
            self.adjancy_matrix.append(temp_list)

        for index, node in enumerate(self.node_list):
            node.subsubgraph_index = index

        for span in self.span_list:
            row = span.get_caller().subsubgraph_index
            col = span.get_callee().subsubgraph_index
            self.adjancy_matrix[row][col] = span
        
        self.get_end_nodes_and_spans()
        self.get_enter_nodes_and_spans()

        self.floyd()
        self.generate_span_chains()

    def floyd(self):
        self.floyd_g = []
        self.floyd_d = []
        nodes_number = len(self.node_list)
        for i in range(0, nodes_number):
            temp_list = [9999]*nodes_number
            self.floyd_g.append(temp_list)
        for i in range(0, nodes_number):
            temp_list = list(range(0, nodes_number))
            self.floyd_d.append(temp_list)
        for row in range(0, nodes_number):
            for col in range(0, nodes_number):
                if self.adjancy_matrix[row][col] != None:
                    self.floyd_g[row][col] = 1

        for k in range(0, nodes_number):
            for i in range(0, nodes_number):
                for j in range(0, nodes_number):
                    if self.floyd_g[i][k] + self.floyd_g[k][j] < self.floyd_g[i][j]:
                        self.floyd_g[i][j] = self.floyd_g[i][k] + self.floyd_g[k][j]
                        self.floyd_d[i][j] = self.floyd_d[i][k]
        
    def generate_span_chains(self):
        span_chains = []
        for enter_node in self.enter_nodes:
            for end_node in self.end_nodes:
                node_chain = [enter_node]
                enter_index = enter_node.subsubgraph_index
                end_index = end_node.subsubgraph_index

                if self.floyd_g[enter_index][end_index] > 2000:
                    print('not reachable')
                    continue

                k = self.floyd_d[enter_index][end_index]
                while k!=end_index:
                    node_chain.append(self.node_list[k])
                    k = self.floyd_d[k][end_index]
                node_chain.append(end_node)
                span_chain = []

                for i in range(0, len(node_chain)):
                    if i+1 < len(node_chain):
                        span_chain.append(self.adjancy_matrix[node_chain[i].subsubgraph_index][node_chain[i+1].subsubgraph_index])
                span_chains.append(span_chain)   

        self.span_chains = span_chains
        return span_chains

    def get_enter_nodes_and_spans(self):
        for col in range(0, len(self.node_list)):
            count = 0
            for row in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][col] != None:
                    count = count + 1
            if count == 0:
                self.enter_nodes.append(self.node_list[col])
        for node in self.enter_nodes:
            for col in range(0, len(self.node_list)):
                if self.adjancy_matrix[node.subsubgraph_index][col] != None:
                    self.enter_spans.append(self.adjancy_matrix[node.subsubgraph_index][col])

    def get_end_nodes_and_spans(self):
        for row in range(0, len(self.node_list)):
            count = 0
            for col in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][col] != None:
                    count = count + 1
            if count == 0:
                self.end_nodes.append(self.node_list[row])
        for node in self.end_nodes:
            for row in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][node.subsubgraph_index] != None:
                    self.end_spans.append(self.adjancy_matrix[row][node.subsubgraph_index])
        


# 子图
class Subgraph:
    def __init__(self, node_list):
        self.span_list = []
        self.node_list = node_list
        self.adjancy_matrix = []

        self.enter_nodes = []
        self.enter_spans = []
        self.end_nodes = []
        self.end_spans = []

        self.span_chains = []

    def construct_matrix(self, adjancy_matrix):
        nodes_number = len(self.node_list)
        for i in range(0, nodes_number):
            temp_list = [None]*nodes_number
            self.adjancy_matrix.append(temp_list)
        
        for index, node in enumerate(self.node_list):
            node.subgraph_index = index

        for row_node in self.node_list:
            for col_node in self.node_list:
                if row_node == col_node:
                    continue
                if adjancy_matrix[row_node.callgraph_index][col_node.callgraph_index] != None:
                    if adjancy_matrix[row_node.callgraph_index][col_node.callgraph_index] not in self.span_list:
                        self.span_list.append(
                        adjancy_matrix[row_node.callgraph_index][col_node.callgraph_index])

        for span in self.span_list:
            row_index = self.node_list.index(span.get_caller())
            col_index = self.node_list.index(span.get_callee())
            self.adjancy_matrix[row_index][col_index] = span

        count = 0
        for row in range(0, len(self.node_list)):
            for col in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][col]!=None:
                    count = count+1
        self.count = count

    def __eq__(self, subgraph):
        return set(self.node_list) == set(subgraph.node_list)

    def get_enter_nodes_and_spans(self):
        for col in range(0, len(self.node_list)):
            count = 0
            for row in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][col] != None:
                    count = count + 1
            if count == 0:
                self.enter_nodes.append(self.node_list[col])
        for node in self.enter_nodes:
            for col in range(0, len(self.node_list)):
                if self.adjancy_matrix[node.subgraph_index][col] != None:
                    self.enter_spans.append(self.adjancy_matrix[node.subgraph_index][col])

    def get_end_nodes_and_spans(self):
        for row in range(0, len(self.node_list)):
            count = 0
            for col in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][col] != None:
                    count = count + 1
            if count == 0:
                self.end_nodes.append(self.node_list[row])
        for node in self.end_nodes:
            for row in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][node.subgraph_index] != None:
                    self.end_spans.append(self.adjancy_matrix[row][node.subgraph_index])

    def generate_span_chains(self):
        self.get_enter_nodes_and_spans()
        self.get_end_nodes_and_spans()
        print('enter span number:',len(self.enter_spans),'end span number', len(self.end_spans))
        print('enter node number:',len(self.enter_nodes),'end node number', len(self.end_nodes))
        sub_subgraph_spans = []
        # 在子图中依据调用关系
        for end_span in self.end_spans:
            sub_subgraph_spans.append(self.get_upstream_spans(end_span))
        for one_subsubgraph_spans in sub_subgraph_spans:
            subsubgraph = SubSubGraph(one_subsubgraph_spans)
            self.span_chains.extend(subsubgraph.span_chains)


    def get_upstream_spans(self, span):
        span_list = [span]
        if span in self.enter_spans:
            return span_list

        col_index = span.get_caller().subgraph_index
        upstream_span_list = []
        for row in range(0, len(self.node_list)):
            if self.adjancy_matrix[row][col_index]!=None:
                upstream_span_list.append(self.adjancy_matrix[row][col_index])
        
        if len(upstream_span_list) > 1:
            real_upstream_span_list = flex_similarity_pruning(span, upstream_span_list)
            for real_upstream_span in real_upstream_span_list:
                span_list.extend(self.get_upstream_spans(real_upstream_span))
        elif len(upstream_span_list) == 1:
            span_list.extend(self.get_upstream_spans(upstream_span_list[0]))
        
        else:
            print('relation error')

        return span_list

# call graph
class Callgraph:
    def __init__(self, all_spans, entry_spans, kind=0):
        self.all_spans = all_spans
        self.entry_spans = entry_spans
        self.kind = kind
        self.subgraphs = []
        self.all_nodes = []
        self.adjancy_matrix = []
        self.generate_all_nodes()
        self.normalize_all_spans()
        self.generate_adjacency_matrix()
        self.generate_all_subgraphs()
        for index, subgraph in enumerate(self.subgraphs):
            print('subgraph -',index,': span number', len(subgraph.span_list),'node number', len(subgraph.node_list), 'matrix count', subgraph.count)
        self.generate_all_span_chains()
        self.root_cause_exploration_in_span_chains(self.kind)

    def normalize_all_spans(self):
        for span in self.all_spans:
            span.normalize_metric()

    def add_sub_graph(self, subgraph):
        self.subgraphs.append(subgraph)

    def generate_all_nodes(self):
        for span in self.all_spans:
            caller_node = Node(*span.get_caller())
            callee_node = Node(*span.get_callee())

            if caller_node not in self.all_nodes:
                self.all_nodes.append(caller_node)
                span.caller = caller_node
            else:
                span.caller = self.all_nodes[self.all_nodes.index(caller_node)]
            if callee_node not in self.all_nodes:
                self.all_nodes.append(callee_node)
                span.callee = callee_node
            else:
                span.callee = self.all_nodes[self.all_nodes.index(callee_node)]

        print('call graph span number', len(self.all_spans))
        print('call graph node number', len(self.all_nodes))

        for index, node in enumerate(self.all_nodes):
            node.callgraph_index = index

    def generate_adjacency_matrix(self):
        nodes_number = len(self.all_nodes)
        for i in range(0, nodes_number):
            temp_list = [None]*nodes_number
            self.adjancy_matrix.append(temp_list)

        for span in self.all_spans:
            temp_caller_node = span.get_caller()
            temp_callee_node = span.get_callee()

            # caller_node_index = self.all_nodes.index(temp_caller_node)
            # callee_node_index = self.all_nodes.index(temp_callee_node)

            caller_node_index = temp_caller_node.callgraph_index
            callee_node_index = temp_callee_node.callgraph_index

            self.adjancy_matrix[caller_node_index][callee_node_index] = span

        count = 0
        for row in range(0, nodes_number):
            for col in range(0, nodes_number):
                if self.adjancy_matrix[row][col]!=None:
                    count = count + 1

        print('matrix span number',count)

    def get_connected_component_by_node(self, node):
        visited = []
        queue = [node]

        while len(queue) != 0:
            top_node = queue[0]
            visited.append(top_node)
            queue.pop(0)

            callgraph_index = top_node.callgraph_index

            for index in range(0, len(self.all_nodes)):
                if self.adjancy_matrix[callgraph_index][index] != None:
                    callee = self.adjancy_matrix[callgraph_index][index].get_callee(
                    )
                    if callee not in visited and callee not in queue:
                        queue.append(callee)
                if self.adjancy_matrix[index][callgraph_index] != None:
                    caller = self.adjancy_matrix[index][callgraph_index].get_caller(
                    )
                    if caller not in visited and caller not in queue:
                        queue.append(caller)

        return visited

    def generate_all_subgraphs(self):
        for node in self.all_nodes:
            temp_node_list = self.get_connected_component_by_node(node)
            temp_subgraph = Subgraph(temp_node_list)
            if temp_subgraph not in self.subgraphs:
                temp_subgraph.construct_matrix(self.adjancy_matrix)
                self.add_sub_graph(temp_subgraph)

    def generate_all_span_chains(self):
        self.span_chains = []
        for subgraph in self.subgraphs:
            subgraph.generate_span_chains()
            self.span_chains.extend(subgraph.span_chains)
            print('span chain number',len(subgraph.span_chains))

    def root_cause_exploration_in_span_chains(self, kind):
        self.root_cause_list = []
        for span_chain in self.span_chains:
            for span in span_chain:
                anomaly_detection_for_one_span(span)
                span.compute_pearsonr_to_entry_span()
        
        if kind == 0:
            for span_chain in self.span_chains:
                span_chain = list(reversed(span_chain))

                start_anomaly_span = None
                start_index = -1
                for index, span in enumerate(span_chain):
                    if span.qpm_anomaly:
                        start_anomaly_span = span
                        start_index = index

                if start_anomaly_span == None:
                    continue
                else:
                    index = start_index
                    while index < len(span_chain):
                        if (index + 1) < len(span_chain) and span_chain[index+1].qpm_anomaly == False:
                            root_cause = Root_Cause(span_chain[index].get_caller().get_turple(), span_chain[index].qpm_beta * span_chain[index].qpm_similarity)
                            if root_cause not in self.root_cause_list:
                                self.root_cause_list.append(root_cause)
                            index = index + 2
                        elif (index + 1) < len(span_chain) and span_chain[index+1].qpm_anomaly == True:
                            index = index + 1
                            root_cause = Root_Cause(span_chain[index+1].get_caller().get_turple(), span_chain[index+1].qpm_beta * span_chain[index+1].qpm_similarity)
                            if root_cause not in self.root_cause_list:
                                self.root_cause_list.append(root_cause)
                        elif (index + 1) == len(span_chain):
                            if span_chain[index].qpm_anomaly == True:
                                root_cause = Root_Cause(span_chain[index].get_caller().get_turple(), span_chain[index].qpm_beta * span_chain[index].qpm_similarity)
                                index = index + 1
                                if root_cause not in self.root_cause_list:
                                    self.root_cause_list.append(root_cause)
                            else:
                                index = index + 1

        elif kind ==1:
            for span_chain in self.span_chains:

                start_anomaly_span = None
                start_index = -1
                for index, span in enumerate(span_chain):
                    if span.ec_anomaly:
                        start_anomaly_span = span
                        start_index = index

                if start_anomaly_span == None:
                    continue
                else:
                    index = start_index
                    while index < len(span_chain):
                        if (index + 1) < len(span_chain) and span_chain[index+1].ec_anomaly == False:
                            root_cause = Root_Cause(span_chain[index].get_callee().get_turple(), span_chain[index].ec_beta * span_chain[index].ec_similarity)
                            if root_cause not in self.root_cause_list:
                                self.root_cause_list.append(root_cause)
                            index = index + 2
                        elif (index + 1) < len(span_chain) and span_chain[index+1].ec_anomaly == True:
                            index = index + 1
                            root_cause = Root_Cause(span_chain[index+1].get_callee().get_turple(), span_chain[index+1].ec_beta * span_chain[index+1].ec_similarity)
                            if root_cause not in self.root_cause_list:
                                self.root_cause_list.append(root_cause)
                        elif (index + 1) == len(span_chain):
                            if span_chain[index].ec_anomaly == True:
                                root_cause = Root_Cause(span_chain[index].get_callee().get_turple(), span_chain[index].ec_beta * span_chain[index].ec_similarity)
                                index = index + 1
                                if root_cause not in self.root_cause_list:
                                    self.root_cause_list.append(root_cause)
                            else:
                                index = index + 1
        elif kind == 2:
            for span_chain in self.span_chains:
    
                start_anomaly_span = None
                start_index = -1
                for index, span in enumerate(span_chain):
                    if span.rt_anomaly:
                        start_anomaly_span = span
                        start_index = index

                if start_anomaly_span == None:
                    continue
                else:
                    index = start_index
                    while index < len(span_chain):
                        if (index + 1) < len(span_chain) and span_chain[index+1].rt_anomaly == False:
                            root_cause = Root_Cause(span_chain[index].get_callee().get_turple(), span_chain[index].rt_beta * span_chain[index].rt_similarity)
                            if root_cause not in self.root_cause_list:
                                self.root_cause_list.append(root_cause)
                            index = index + 2
                        elif (index + 1) < len(span_chain) and span_chain[index+1].rt_anomaly == True:
                            index = index + 1
                            root_cause = Root_Cause(span_chain[index+1].get_callee().get_turple(), span_chain[index+1].rt_beta * span_chain[index+1].rt_similarity)
                            if root_cause not in self.root_cause_list:
                                self.root_cause_list.append(root_cause)
                        elif (index + 1) == len(span_chain):
                            if span_chain[index].rt_anomaly == True:
                                root_cause = Root_Cause(span_chain[index].get_callee().get_turple(), span_chain[index].rt_beta * span_chain[index].rt_similarity)
                                index = index + 1
                                if root_cause not in self.root_cause_list:
                                    self.root_cause_list.append(root_cause)
                            else:
                                index = index + 1
        self.root_cause_list = sorted(self.root_cause_list, key=lambda x: x.root_score, reverse=True)
        for root_cause in self.root_cause_list:
            print(root_cause)

def anomaly_detection_for_one_span(span):
    caller_turple = span.get_caller()
    callee_turple = span.get_callee()

    span_day_before = None
    span_7days_before = None

    metric1 = None
    metric7 = None

    caller_callee_turple = caller_turple + callee_turple
    if caller_callee1.get(caller_callee_turple) != None:
        metric1 = caller_callee1[caller_callee_turple]
    if caller_callee7.get(caller_callee_turple) != None:
        metric7 = caller_callee7[caller_callee_turple]

    span_day_before = Span(caller_turple, callee_turple, metric1)
    span_7days_before = Span(caller_turple, callee_turple, metric7)
    span_day_before.normalize_metric()
    span_7days_before.normalize_metric()

    exceptions_qpm = [-1] * 30 
    exceptions_ec = [-1] * 30
    exceptions_rt = [-1] * 30

    current_qpm = None
    current_ec = None
    current_rt = None
    
    if alarm_time >= 30:
        current_qpm = span.qpm[alarm_time-30: alarm_time]
        current_ec = span.ec[alarm_time-30: alarm_time]
        current_rt = span.rt[alarm_time-30: alarm_time]
    else:
        current_qpm = span.qpm[0:alarm_time]
        current_ec = span.ec[0:alarm_time]
        current_rt = span.rt[0:alarm_time]

    # TODO 添加条件判断, alarm_time早于 01:00, 取前一天数据补足
    comparison_qpm_today = span.qpm[alarm_time-60: alarm_time]
    comparison_qpm_day_before = span_day_before.qpm[alarm_time-60: alarm_time]
    comparison_qpm_7days_before = span_7days_before.qpm[alarm_time-60: alarm_time]

    comparison_ec_today = span.ec[alarm_time-60: alarm_time]
    comparison_ec_day_before = span_day_before.ec[alarm_time-60: alarm_time]
    comparison_ec_7days_before = span_7days_before.ec[alarm_time-60: alarm_time]

    comparison_rt_today = span.rt[alarm_time-60: alarm_time]
    comparison_rt_day_before = span_day_before.rt[alarm_time-60: alarm_time]
    comparison_rt_7days_before = span_7days_before.rt[alarm_time-60: alarm_time]

    sigma(exceptions_qpm, comparison_qpm_today, current_qpm, 0)
    sigma(exceptions_qpm, comparison_qpm_day_before, current_qpm, 0)
    sigma(exceptions_qpm, comparison_qpm_7days_before, current_qpm, 0)

    sigma(exceptions_ec, comparison_ec_today, current_ec, 1)
    sigma(exceptions_ec, comparison_ec_day_before, current_ec, 1)
    sigma(exceptions_ec, comparison_ec_7days_before, current_ec, 1)

    sigma(exceptions_rt, comparison_rt_today, current_rt, 2)
    sigma(exceptions_rt, comparison_rt_day_before, current_rt, 2)
    sigma(exceptions_rt, comparison_rt_7days_before, current_rt, 2)

    qpm_valid = 30 - exceptions_qpm.count(-1)
    ec_valid = 30 - exceptions_ec.count(-1)
    rt_valid = 30 - exceptions_rt.count(-1)

    if qpm_valid != 0:
        if 1.0*exceptions_qpm.count(1) / (30-exceptions_qpm.count(-1)) > threshold:
            span.qpm_anomaly = True
            span.qpm_beta = 1.0*exceptions_qpm.count(1) / (30-exceptions_qpm.count(-1))
    
    if ec_valid != 0:
        if 1.0*exceptions_ec.count(1) / (30-exceptions_ec.count(-1)) > threshold:
            span.ec_anomaly = True
            span.ec_beta = 1.0*exceptions_ec.count(1) / (30-exceptions_ec.count(-1))

    if rt_valid != 0:
        if 1.0*exceptions_rt.count(1) / (30-exceptions_rt.count(-1)) > threshold:
            span.rt_anomaly = True
            span.rt_beta = 1.0*exceptions_rt.count(1) / (30-exceptions_rt.count(-1))
            

def data_valid(data_list):
    count = 0
    for point in data_list:
        if not np.isnan(point):
            count = count + 1
    return count > 5

def sigma(exceptions, comparison, current, kind):
    if (not data_valid(comparison)) or (not data_valid(current)):
        # print('not enough data')
        return
    mean, std, count = calculate(comparison)
    min = mean - 3*std
    max = mean + 3*std
    for index, date_point in enumerate(current):
         if not np.isnan(current[index]):
                exceptions[index] = 0
                if kind == 0:
                    if current[index]< min or current[index] > max:
                        exceptions[index] = 1
                else:
                    if current[index] > max:
                        exceptions[index] = 1

def calculate(cal_data):
    count=0
    sum=0.0
    std=0.0
    for i in cal_data:
        if not np.isnan(i):
            count+=1
            sum+=i
    mean=1.0*sum/count
    for i in cal_data:
        if not  np.isnan(i):
            std+=(i-mean)**2
    std=(std/count)**0.5
    return mean,std,count

def split_data(data):
    time_list = [np.nan]*1440
    data_points = data.split(',')

    for data_point in data_points:
        time_list[int(data_point.split(':')[0])] = float(data_point.split(':')[1])

    return time_list

# 遇到分支需要流量相似度分辨调用关系
def flex_similarity_pruning(downstream_span, upstream_spans):
    choose_list = []
    for span in upstream_spans:
        choose_list.append((span,))
    if len(upstream_spans)>1:
        for i in range(2, len(upstream_spans)+1):
            
            choose_list.extend(list(it.combinations(upstream_spans, i)))

    pruning_list = []
    for spans in choose_list:
        pruning = Pearsonr_Pruning(downstream_span, list(spans))
        pruning_list.append(pruning)

    pruning_list = sorted(pruning_list, key = lambda x:x.similarity)
    return pruning_list[-1].compare_span_list 
    


def merge_dict(dic1, dic2):
    for key, value in dic2.items():
        if key in dic1:
            dic1[key].extend(value)
        else:
            dic1[key] = value


def read_one_file(lock, path, file, caller_data_local, callee_data_local, caller_callee_local):
    thread_caller_data = {}
    thread_callee_data = {}
    thread_caller_callee = {}

    if file.startswith('part'):
        file_path = path+file
        temp_file = pd.read_csv(file_path, sep='|')
        for i, line in temp_file.iterrows():
            cur_node = (line[1], line[2], line[3], line[4],
                        line[5], line[6], line[7], line[8])
            if thread_caller_data.get((line[1], line[2], line[3], line[4])) == None:
                thread_caller_data[(
                    line[1], line[2], line[3], line[4])] = list()
            thread_caller_data[(line[1], line[2], line[3],
                                line[4])].append(cur_node)

            if thread_callee_data.get((line[5], line[6], line[7], line[8])) == None:
                thread_callee_data[(
                    line[5], line[6], line[7], line[8])] = list()
            thread_callee_data[(line[5], line[6], line[7],
                                line[8])].append(cur_node)

            thread_caller_callee[cur_node] = [
                line[9], line[10], line[12], line[13]]
    lock.acquire()
    merge_dict(caller_data_local, thread_caller_data)
    merge_dict(callee_data_local, thread_callee_data)
    merge_dict(caller_callee_local, thread_caller_callee)
    lock.release()


def read_files(path):
    temp_caller_data = {}
    temp_callee_data = {}
    temp_caller_callee = {}

    files = os.listdir(path)
    threads_local = []
    lock = threading.Lock()
    for file in files:
        thread = threading.Thread(target=read_one_file, args=(
            lock, path, file, temp_caller_data, temp_callee_data, temp_caller_callee))
        threads_local.append(thread)
        thread.start()

    for thread in threads_local:
        thread.join()

    return (temp_caller_data, temp_callee_data, temp_caller_callee)


def get_uplink_spans(span, entry_span):
    temp_spans = []
    if callee_data.get(span.get_caller()) != None:
        for caller_callee_turple in callee_data[span.get_caller()]:
            caller = (caller_callee_turple[0], caller_callee_turple[1],
                      caller_callee_turple[2], caller_callee_turple[3])
            callee = (caller_callee_turple[4], caller_callee_turple[5],
                      caller_callee_turple[6], caller_callee_turple[7])
            metric = caller_callee[caller_callee_turple]
            new_span = Span(caller, callee, metric, entry_span=entry_span)
            # if (new_span not in temp_spans) and (new_span not in uplink_cache_spans):
            temp_spans.append(new_span)
    return temp_spans


def get_downlink_spans(span, entry_span):
    temp_spans = []
    if caller_data.get(span.get_callee()) != None:
        for caller_callee_turple in caller_data[span.get_callee()]:
            caller = (caller_callee_turple[0], caller_callee_turple[1],
                      caller_callee_turple[2], caller_callee_turple[3])
            callee = (caller_callee_turple[4], caller_callee_turple[5],
                      caller_callee_turple[6], caller_callee_turple[7])
            metric = caller_callee[caller_callee_turple]

            new_span = Span(caller, callee, metric, entry_span=entry_span)
            # if (new_span not in temp_spans) and (new_span not in downlink_cache_spans):
            temp_spans.append(new_span)
    return temp_spans

def extend_anomaly_entry_span(caller_spans, callee_spans):
    uplink_cache_spans = []
    downlink_cache_spans = []

    for span in caller_spans:
        downlink_cache_spans.extend(get_downlink_spans(span, span))
    for span in callee_spans:
        uplink_cache_spans.extend(get_uplink_spans(span, span))

    while len(uplink_cache_spans) != 0:
        top_span = uplink_cache_spans[0]
        uplink_cache_spans.pop(0)
        uplink_spans.append(top_span)

        uplink_cache_spans.extend(get_uplink_spans(top_span, top_span.entry_span))

    while len(downlink_cache_spans) != 0:
        top_span = downlink_cache_spans[0]
        downlink_cache_spans.pop(0)
        downlink_spans.append(top_span)

        downlink_cache_spans.extend(get_downlink_spans(top_span, top_span.entry_span))

    all_related_spans = []
    entry_spans = []

    entry_spans.extend(caller_spans)
    entry_spans.extend(callee_spans)

    all_related_spans.extend(entry_spans)
    all_related_spans.extend(uplink_spans)
    all_related_spans.extend(downlink_spans)

    return (entry_spans, list(set(all_related_spans)))


if __name__ == '__main__':

    item_server = 'caccfe3f3052dab10bceb9912fe43645'
    # item_server = 'de47d06de87681602a1ff5fcd05b30e2'
    alarm_times = '2021/12/18 17:53'.split()[1].split(':')
    alarm_time = int(alarm_times[0]) * 60 + int(alarm_times[1])
    alarm_date = datetime.datetime.strptime('20211218', '%Y%m%d')

    threshold = 0.2

    # 多进程并行计算
    process_pool = mp.Pool(processes=3)

    data_read_results = []

    path_today = './app_opsdatagovern_aiops_export_caller_min_monitor_di/' + \
        alarm_date.strftime('%Y%m%d') + '/'
    path_day_before = './app_opsdatagovern_aiops_export_caller_min_monitor_di/' + \
        (alarm_date-datetime.timedelta(days=1)).strftime('%Y%m%d') + '/'
    path_7days_before = './app_opsdatagovern_aiops_export_caller_min_monitor_di/' + \
        (alarm_date-datetime.timedelta(days=7)).strftime('%Y%m%d') + '/'

    start_time = time.time()

    data_read_results.append(process_pool.apply_async(
        read_files, args=(path_today, )))
    data_read_results.append(process_pool.apply_async(
        read_files, args=(path_day_before, )))
    data_read_results.append(process_pool.apply_async(
        read_files, args=(path_7days_before, )))

    data_read_results = [data.get() for data in data_read_results]

    caller_data = data_read_results[0][0]
    callee_data = data_read_results[0][1]
    caller_callee = data_read_results[0][2]

    caller_data1 = data_read_results[1][0]
    callee_data1 = data_read_results[1][1]
    caller_callee1 = data_read_results[1][2]

    caller_data7 = data_read_results[2][0]
    callee_data7 = data_read_results[2][1]
    caller_callee7 = data_read_results[2][2]

    del data_read_results

    print('数据读取耗时: ', time.time() - start_time)

    uplink_spans = []
    downlink_spans = []
    entry_spans_as_callee = []
    entry_spans_as_caller = []

    entry_qpm_anomaly_span_as_caller = []
    entry_qpm_anomaly_span_as_callee = []

    entry_ec_anomaly_span_as_caller = []
    entry_ec_anomaly_span_as_callee = []

    entry_rt_anomaly_span_as_caller = []
    entry_rt_anomaly_span_as_callee = []

    for key, caller_callee_turples in caller_data.items():
        if key[0] == item_server:
            for caller_callee_turple in caller_callee_turples:
                caller = (caller_callee_turple[0], caller_callee_turple[1],
                          caller_callee_turple[2], caller_callee_turple[3])
                callee = (caller_callee_turple[4], caller_callee_turple[5],
                          caller_callee_turple[6], caller_callee_turple[7])
                metric = caller_callee[caller_callee_turple]
                span = Span(caller, callee, metric, is_entry=True)
                entry_spans_as_caller.append(span)
        else:
            continue

    for key, caller_callee_turples in callee_data.items():
        if key[0] == item_server:
            for caller_callee_turple in caller_callee_turples:
                caller = (caller_callee_turple[0], caller_callee_turple[1],
                          caller_callee_turple[2], caller_callee_turple[3])
                callee = (caller_callee_turple[4], caller_callee_turple[5],
                          caller_callee_turple[6], caller_callee_turple[7])
                metric = caller_callee[caller_callee_turple]
                span = Span(caller, callee, metric, is_entry=True)
                entry_spans_as_callee.append(span)
        else:
            continue

    # 筛选与报警item直接相关的调用记录
    # if caller_data.get(item_server)!=None:
    #     for caller_callee_turple in caller_data[item_server]:
    #         caller = (caller_callee_turple[0], caller_callee_turple[1],
    #                     caller_callee_turple[2], caller_callee_turple[3])
    #         callee = (caller_callee_turple[4], caller_callee_turple[5],
    #                     caller_callee_turple[6], caller_callee_turple[7])
    #         metric = caller_callee[caller_callee_turple]
    #         span = Span(caller, callee, metric)
    #         entry_spans_as_caller.append(span)

    # if callee_data.get(item_server)!=None:
    #     for caller_callee_turple in callee_data[item_server]:
    #         caller = (caller_callee_turple[0], caller_callee_turple[1],
    #                     caller_callee_turple[2], caller_callee_turple[3])
    #         callee = (caller_callee_turple[4], caller_callee_turple[5],
    #                     caller_callee_turple[6], caller_callee_turple[7])
    #         metric = caller_callee[caller_callee_turple]
    #         span = Span(caller, callee, metric)
    #         entry_spans_as_callee.append(span)

    print('direct spans finished', len(entry_spans_as_caller), len(entry_spans_as_callee))

    # 对入口的span进行异常检测, 分类别加入相应的列表
    for span in entry_spans_as_caller:
        span.normalize_metric()
        anomaly_detection_for_one_span(span)
        if span.qpm_anomaly:
            entry_qpm_anomaly_span_as_caller.append(span)
        if span.ec_anomaly:
            entry_ec_anomaly_span_as_caller.append(span)
        if span.rt_anomaly:
            entry_rt_anomaly_span_as_caller.append(span)

    for span in entry_spans_as_callee:
        span.normalize_metric()
        anomaly_detection_for_one_span(span)
        if span.qpm_anomaly:
            entry_qpm_anomaly_span_as_callee.append(span)
        if span.ec_anomaly:
            entry_ec_anomaly_span_as_callee.append(span)
        if span.rt_anomaly:
            entry_rt_anomaly_span_as_callee.append(span)

    print('qpm anomaly entry number', len(entry_qpm_anomaly_span_as_caller)+len(entry_qpm_anomaly_span_as_callee))
    print('ec anomaly entry number', len(entry_ec_anomaly_span_as_caller)+len(entry_ec_anomaly_span_as_callee))
    print('rt anomaly entry number', len(entry_rt_anomaly_span_as_caller)+len(entry_rt_anomaly_span_as_callee))

    # 根据上面筛选到的入口span, 分别向上游和下游拓展相关span

    # uplink_cache_spans = []
    # downlink_cache_spans = []

    # for span in entry_spans_as_caller:
    #     downlink_cache_spans.extend(get_downlink_spans(span))
    # for span in entry_spans_as_callee:
    #     uplink_cache_spans.extend(get_uplink_spans(span))

    # while len(uplink_cache_spans) != 0:
    #     top_span = uplink_cache_spans[0]
    #     uplink_cache_spans.pop(0)
    #     uplink_spans.append(top_span)

    #     uplink_cache_spans.extend(get_uplink_spans(top_span))

    # while len(downlink_cache_spans) != 0:
    #     top_span = downlink_cache_spans[0]
    #     downlink_cache_spans.pop(0)
    #     downlink_spans.append(top_span)

    #     downlink_cache_spans.extend(get_downlink_spans(top_span))

    # print('spans拓展完毕', len(uplink_spans), len(downlink_spans))

    # all_related_spans = []
    # entry_spans = []

    # entry_spans.extend(entry_spans_as_caller)
    # entry_spans.extend(entry_spans_as_callee)

    # all_related_spans.extend(entry_spans)
    # all_related_spans.extend(uplink_spans)
    # all_related_spans.extend(downlink_spans)

    # print('all related spans:', len(list(set(all_related_spans))))

    # 将拓展转移到函数中
    qpm_anomaly_entry_spans, qpm_anomaly_all_spans = extend_anomaly_entry_span(entry_qpm_anomaly_span_as_caller, entry_qpm_anomaly_span_as_callee)
    ec_anomaly_entry_spans, ec_anomaly_all_spans = extend_anomaly_entry_span(entry_ec_anomaly_span_as_caller, entry_ec_anomaly_span_as_callee)
    rt_anomaly_entry_spans, rt_anomaly_all_spans = extend_anomaly_entry_span(entry_rt_anomaly_span_as_caller, entry_rt_anomaly_span_as_callee)
 
    if len(qpm_anomaly_all_spans) == 0:
        print('no qpm anomaly or lack of data')
    else:
        qpm_callgraph = Callgraph(qpm_anomaly_all_spans, qpm_anomaly_entry_spans, 0)
    if len(ec_anomaly_all_spans) == 0:
        print('no ec anomaly or lack of data')
    else:
        ec_callgraph = Callgraph(ec_anomaly_all_spans, ec_anomaly_entry_spans, 1)
    if len(rt_anomaly_all_spans) == 0:
        print('no rt anomaly or lack of data')
    else:
        rt_callgraph = Callgraph(rt_anomaly_all_spans, rt_anomaly_entry_spans, 2)