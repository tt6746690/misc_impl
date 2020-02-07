from graphviz import Digraph


def save_graph(filename,nodes,edges,labels):

    dot = Digraph(format='png',filename=filename,graph_attr={'rankdir':'LR','dpi': '250'})

    dot.attr('node', shape='circle')
    for i in nodes:
        dot.node(str(i),str(i))

    for (u,v),label in zip(edges,labels):
        dot.edge(str(u),str(v),label=label)

    dot.render()


nodes = ['c', 'x']
edges = [('c', 'x')]
labels = ['']

save_graph(f'./graph',nodes,edges,labels)