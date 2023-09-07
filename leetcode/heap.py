"""
heap
- a tree that satisfies the heap property. in a max heap, value of any node is larger than that of its children's
- a data structure used to implement abstract data type called priority queue


binary heap: https://en.wikipedia.org/wiki/Binary_heap
find-max: O(1)
insert/push: O(log n)
remove root: O(log n)
heapify: O(n)
"""


if __name__ == '__main__':
    from queue import PriorityQueue

    # min-heap
    q = PriorityQueue()

    # first item is priority
    q.put((4, 'Read'))
    q.put((2, 'Play'))
    q.put((5, 'Write'))
    q.put((1, 'Code'))
    q.put((3, 'Study'))

    while not q.empty(): # need to use `.empty()`!
        # heappop max element
        next_item = q.get()
        print(next_item)