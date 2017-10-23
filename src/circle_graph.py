'''
Holds the a basic graph structure and methods to perform common
graph algorithm tasks. Does not handle multiple node types
'''

import pickle
import time
from collections import Counter
import networkx as nx
import pandas as pd


class CircleGraph(object):
    '''
    Generic container for a directed graph allowing for parallel edges
    '''

    def __init__(self, graph=None, g_type='multi'):
        '''
        Initializes object to have a MultiDiGraph
        '''
        if graph is None:
            if g_type is 'multi':
                self.graph = nx.MultiDiGraph()
            else:
                self.graph = nx.MultiGraph()
        else:
            self.graph = graph

    def populate_graph(self, data):
        '''
        Populates the graph with either weighted or unweighted edges.
        Continues to be a directed graph.
        '''
        if data.shape[1] == 2:
            self.graph.add_edges_from(data)
        elif data.shape[1] == 3:
            self.graph.add_weighted_edges_from(data)
        else:
            raise Exception

    def largest_component(self):
        '''
        Returns the largest weakly connected component subgraph
        '''
        g = max(nx.weakly_connected_component_subgraphs(self.graph),
                key=len)
        return CircleGraph(g)

    def get_degree_centrality(self):
        '''
        Returns the degree centrality for every node in the graph
        '''
        return nx.degree_centrality(self.graph)

    def get_closeness_centrality(self):
        '''
        Returns the closeness centrality for every node in the graph
        '''
        return nx.closeness_centrality(self.graph)

    def get_betweenness_centrality(self):
        '''
        Returns the betweenness centrality for every node in
        the graph
        '''
        return nx.betweenness_centrality(self.graph)


def max_n_key_value(d, n):
    '''
    Returns the key:value pair of the max n items using the value
    as a key
    '''
    return Counter(d).most_common(n)


def snowball_sample(g, center, max_depth=1, current_depth=0,
                    marked=[]):
    '''
    Samples from a given center point by recursively looking
    at connections of connections
    '''

    # Terminal condition for max_depth
    if current_depth == max_depth:
        return marked

    # Terminal condition for already visited node
    if center in marked:
        return marked

    # Recursively call in a BFS style
    else:
        marked.append(center)

    if g[center].keys() is not None:
        for node in g[center].keys():
            marked = snowball_sample(
                g,
                node,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                marked=marked)
    return marked

if __name__ == '__main__':
    RUN = False
    DATA_DIR = '../data/'
    if RUN:
        df = pd.read_csv(DATA_DIR + 'uk_spends_02.csv')
        og = CircleGraph()
        og.populate_graph(df.values[:, :2])

        g = og.largest_component()
        pickle.dump(g, open(DATA_DIR + 'largest_component.p', 'wb'))

        start = time.time()
        dc = g.get_degree_centrality()
        pickle.dump(dc, open(DATA_DIR + 'degree_centrality.p', 'wb'))
        end = time.time()
        print "Computed Degree Centrality in %.3f" % (end - start)

        start = time.time()
        cc = g.get_closeness_centrality()
        pickle.dump(cc, open(DATA_DIR + 'closeness_centrality.p', 'wb'))
        end = time.time()
        print "Compted Closeness Centrality in %.3f" % (end - start)

        start = time.time()
        bc = g.get_betweenness_centrality()
        pickle.dump(bc, open(DATA_DIR + 'betweenness_centrality.p', 'wb'))
        end = time.time()
        print "Computed Betweeness Centrality in %.3f" % (end - start)

    g = pickle.load(open(DATA_DIR + 'largest_component.p', 'rb'))
    dc = pickle.load(open(DATA_DIR + 'degree_centrality.p', 'rb'))
    cc = pickle.load(open(DATA_DIR + 'closeness_centrality.p', 'rb'))
    bc = pickle.load(open(DATA_DIR + 'betweenness_centrality.p', 'rb'))

    print max_n_key_value(dc, 5)
    print '\n'
    print max_n_key_value(cc, 5)
    print '\n'
    print max_n_key_value(bc, 5)
