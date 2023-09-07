"""
A topological sort is a graph traversal in which each node v is visited only after all its dependencies are visited. 
A topological sort is only possible if the graph has no directed cycles, i.e. it is a directed acyclic graph (DAG).
Any DAG has at least one topological sort.

Time Complexity: O(V+E)


Kahn's Algorithm

L ← Empty list that will contain the sorted elements
S ← Set of all nodes with no incoming edge
while S is non-empty do
    remove a node n from S
    add n to L
    for each node m with an edge e from n to m do
        remove edge e from the graph
        if m has no other incoming edges then
            insert m into S
if graph has edges then
    return error (graph has at least one cycle)
else
    return L (a topologically sorted order)

"""

