# -*- coding: UTF-8 -*-
from __future__ import print_function
from ast import Sub
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
import time
import datetime
import threading
import multiprocessing as mp

# 完善了多进程+多线程数据加载
# 保留前一版本的caller, callee, caller_callee以方便查找(反正内存够用)
# 经过反复修改, 将span作为代码中最根本的数据单元
# call graph 中的边


class Span:
    # caller, callee 类型为Node
    def __init__(self, caller, callee, metric):
        self.caller = caller
        self.callee = callee
        self.metric = metric

        self.discovered = False
        self.anomaly_detected = False

        self.callgraph_index = -1

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

    def get_turple(self):
        return (self.server, self.service, self.method, self.set)

# 子图中的一条调用链


class Span_Chain:

    def __init__(self):
        self.span_list = []

    def add_span_to_top(self, span):
        self.span_list.insert(0, span)

    def add_span_to_bottom(self, span):
        self.span_list.append(span)


# 子图
class Subgraph:
    def __init__(self, node_list):
        self.span_list = []
        self.node_list = node_list
        self.adjancy_matrix = []

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


# call graph
class Callgraph:
    def __init__(self, all_spans, entry_spans):
        self.all_spans = all_spans
        self.entry_spans = entry_spans
        self.subgraphs = []
        self.all_nodes = []
        self.adjancy_matrix = []
        self.generate_all_nodes()
        self.generate_adjacency_matrix()
        self.generate_all_subgraphs()

        for index, subgraph in enumerate(self.subgraphs):
            print('subgraph -',index,': span number', len(subgraph.span_list),'node number', len(subgraph.node_list), 'matrix count', subgraph.count)

        self.generate_span_chains()


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

    def generate_span_chains(self):
        pass

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


def get_uplink_spans(span):
    temp_spans = []
    if callee_data.get(span.get_caller()) != None:
        for caller_callee_turple in callee_data[span.get_caller()]:
            caller = (caller_callee_turple[0], caller_callee_turple[1],
                      caller_callee_turple[2], caller_callee_turple[3])
            callee = (caller_callee_turple[4], caller_callee_turple[5],
                      caller_callee_turple[6], caller_callee_turple[7])
            metric = caller_callee[caller_callee_turple]
            new_span = Span(caller, callee, metric)
            # if (new_span not in temp_spans) and (new_span not in uplink_cache_spans):
            temp_spans.append(new_span)
    return temp_spans


def get_downlink_spans(span):
    temp_spans = []
    if caller_data.get(span.get_callee()) != None:
        for caller_callee_turple in caller_data[span.get_callee()]:
            caller = (caller_callee_turple[0], caller_callee_turple[1],
                      caller_callee_turple[2], caller_callee_turple[3])
            callee = (caller_callee_turple[4], caller_callee_turple[5],
                      caller_callee_turple[6], caller_callee_turple[7])
            metric = caller_callee[caller_callee_turple]

            new_span = Span(caller, callee, metric)
            # if (new_span not in temp_spans) and (new_span not in downlink_cache_spans):
            temp_spans.append(new_span)
    return temp_spans


if __name__ == '__main__':

    item_server = 'caccfe3f3052dab10bceb9912fe43645'
    # item_server = 'de47d06de87681602a1ff5fcd05b30e2'
    alarm_times = '2021/12/18 17:53'.split()[1].split(':')
    alarm_time = int(alarm_times[0]) * 60 + int(alarm_times[1])
    alarm_date = datetime.datetime.strptime('20211218', '%Y%m%d')

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

    for key, caller_callee_turples in caller_data.items():
        if key[0] == item_server:
            for caller_callee_turple in caller_callee_turples:
                caller = (caller_callee_turple[0], caller_callee_turple[1],
                          caller_callee_turple[2], caller_callee_turple[3])
                callee = (caller_callee_turple[4], caller_callee_turple[5],
                          caller_callee_turple[6], caller_callee_turple[7])
                metric = caller_callee[caller_callee_turple]
                span = Span(caller, callee, metric)
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
                span = Span(caller, callee, metric)
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
    print('direct spans finished', len(
        entry_spans_as_caller), len(entry_spans_as_callee))
    # 根据上面筛选到的入口span, 分别向上游和下游拓展相关span

    uplink_cache_spans = []
    downlink_cache_spans = []

    for span in entry_spans_as_caller:
        downlink_cache_spans.extend(get_downlink_spans(span))
    for span in entry_spans_as_callee:
        uplink_cache_spans.extend(get_uplink_spans(span))

    while len(uplink_cache_spans) != 0:
        top_span = uplink_cache_spans[0]
        uplink_cache_spans.pop(0)
        uplink_spans.append(top_span)

        uplink_cache_spans.extend(get_uplink_spans(top_span))

    while len(downlink_cache_spans) != 0:
        top_span = downlink_cache_spans[0]
        downlink_cache_spans.pop(0)
        downlink_spans.append(top_span)

        downlink_cache_spans.extend(get_downlink_spans(top_span))

    print('spans拓展完毕', len(uplink_spans), len(downlink_spans))

    all_related_spans = []
    entry_spans = []

    entry_spans.extend(entry_spans_as_caller)
    entry_spans.extend(entry_spans_as_callee)

    all_related_spans.extend(entry_spans)
    all_related_spans.extend(uplink_spans)
    all_related_spans.extend(downlink_spans)

    print('all related spans:', len(list(set(all_related_spans))))

    callgraph = Callgraph(list(set(all_related_spans)), entry_spans)
