"""
bfs
- it uses a queue (First In First Out)
- it checks whether a vertex has been explored before enqueueing the vertex rather than delaying this check until the vertex is dequeued from the queue.


```
procedure BFS(G, root) is
    let Q be a queue
    label root as explored
    Q.enqueue(root)
    while Q is not empty do
        v := Q.dequeue()
        if v is the goal then
            return v
        for all edges from v to w in G.adjacentEdges(v) do
            if w is not labeled as explored then
                label w as explored
                w.parent := v
                Q.enqueue(w)
```
dfs
- it uses a stack
- it delays checking whether a vertex has been discovered until the vertex is popped from the stack rather than making this check before adding the vertex.
    - implying that there is the possibility of duplicate vertices on the stack, with worst-case space complexity of O(|E|)

```
procedure DFS_iterative(G, v) is
    let S be a stack
    S.push(v)
    while S is not empty do
        v = S.pop()
        if v is not labeled as discovered then
            label v as discovered
            for all edges from v to w in G.adjacentEdges(v) do 
                S.push(w)
```

https://11011110.github.io/blog/2013/12/17/stack-based-graph-traversal.html
- Stack-based graph traversal ≠ depth first search

- bfs more suitable for finding shortest path, while dfs is more suitable for finding longest path
- bfs is more predictable
- can replace list with `deque` implementation. 
    ```
    from collections import deque
    # queue: popleft() and append()
    # stack: pop()     and append()
    ```
"""


def bfs(adj_list, root):
    """ bfs iterative
        Time:  O(|V|+|E|)
        Space: O(|V|)
    """
    Q = []
    visited = set()
    Q.append(root)
    visited.add(root)
    while Q:
        u = Q.pop(0)
        print(f'node={u}; queue={Q}')
        for v in adj_list[u]:
            if v not in visited:
                visited.add(v)
                Q.append(v)


def dfs(adj_list, root):
    """ dfs iterative
        Time:  O(|V|+|E|)
        Space: O(|E|)
        extra space due to possibly duplicate vertices on the stack.
    """
    S = []
    visited = set()
    S.append(root)
    while S:
        u = S.pop()
        if u not in visited:
            print(f'node={u}; stack={S}')
            visited.add(u)
            for v in adj_list[u]:
                S.append(v)



if __name__ == '__main__':

    # https://en.wikipedia.org/wiki/Depth-first_search
    adj_list = {
        'A': ['B','C','E'],
        'B': ['A','D','F'],
        'C': ['A','G'],
        'D': ['B'],
        'F': ['B','E'],
        'E': ['A','F'],
        'G': ['C'],
    }

    print('bfs:')
    bfs(adj_list, 'A')
    # node=A; queue=[]
    # node=B; queue=['C', 'E']
    # node=C; queue=['E', 'D', 'F']
    # node=E; queue=['D', 'F', 'G']
    # node=D; queue=['F', 'G']
    # node=F; queue=['G']
    # node=G; queue=[]
    print('dfs:')
    dfs(adj_list, 'A')
    # dfs:
    # node=A; stack=[]
    # node=E; stack=['B', 'C']
    # node=F; stack=['B', 'C', 'A']
    # node=B; stack=['B', 'C', 'A']
    # node=D; stack=['B', 'C', 'A', 'A']
    # node=C; stack=['B']
    # node=G; stack=['B', 'A']