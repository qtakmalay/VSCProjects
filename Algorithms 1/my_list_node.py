class MyListNode:
    def __init__(self, elem: int, prev_node: 'MyListNode' = None, next_node: 'MyListNode' = None):
        self.elem = elem
        self.next_node = next_node
        self.prev_node = prev_node
