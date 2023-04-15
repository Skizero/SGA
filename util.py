from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.datasets import Actor, WebKB, WikipediaNetwork, Planetoid

actor = ('actor',)
wikipedia = ('chameleon', 'squirrel')
webkb = ('Cornell', 'Texas', 'Wisconsin')
citation = ('Cora', 'Citeseer', 'PubMed')

def load_graph(name):
    if name in actor:
        data = Actor(root='./dataset', transform=NormalizeFeatures())
    elif name in webkb:
        data = WebKB(root='./dataset', name=name, transform=NormalizeFeatures())
    elif name in wikipedia:
        data = WikipediaNetwork(root='./dataset', name=name, transform=NormalizeFeatures())
    elif name in citation:
        data = Planetoid(root='./dataset', name=name, split='geom-gcn', transform=NormalizeFeatures())

    g = data[0]

    g.F = data.num_node_features
    g.C = data.num_classes

    g.edge_index = remove_self_loops(g.edge_index)[0]
    g.edge_index = to_undirected(g.edge_index)
    return g 
