from typing import Optional

class MinHeap:
    def __init__(self):
        self.heap = []
        self.size = len(self.heap)

    def get_size(self) -> int:
        """
        @return number of elements in the min heap
        """
        return len(self.heap)

    def is_empty(self) -> bool:
        """
        @return True if the min heap is empty, False otherwise
        """
        empty = True if len(self.heap) == 0 else False
        return empty
        pass

    def insert(self, integer_val: int) -> None:
        """
        inserts integer_val into the min heap
        @param integer_val: the value to be inserted
        @raises ValueError if integer_val is None or not an int
        """
        if integer_val == None or not isinstance(integer_val, int):
            raise ValueError

        self.heap.append(integer_val)
        node = len(self.heap) - 1

        while integer_val < self.heap[self.parent(node)]:
            if self.parent(node) < 0:
                break
            else:
                self.swap(self.parent(node), node)
                node = self.parent(node)

    def get_min(self) -> Optional[int]:
        """
        returns the value of the minimum element of the PQ without removing it
        @return the minimum value of the PQ or None if no element exists
        """
        if not self.is_empty():
            return self.heap[0]
        else:
            return None

    def remove_min(self) -> Optional[int]:
        """
        removes the minimum element from the PQ and returns its value
        @return the value of the removed element or None if no element exists
        """
        if self.is_empty():
            return None

        min_val = self.down_heap()
        print(min_val)
        return min_val


    def up_heap(self, index):
        node = self.get_size() - 1
        parent = self.parent(node)
        while index < self.heap[parent]:
            self.swap(parent, node)
            node = self.parent(node)

    @property
    def down_heap(self, node):
        len_heap = len(self.heap)
        while True:
            rnode = self.right_child(node)
            lnode = self.left_child(node)
            if (lnode < len_heap) and (rnode < len_heap):
                if self.heap[lnode] <= self.heap[rnode]:
                    m = lnode
                else:
                    m = rnode
                if self.heap[node] > self.heap[m]:
                    self.swap(m, node)
                    node = m
            elif lnode < len_heap:
                if self.heap[node] > self.heap[lnode]:
                    self.swap(lnode, node)
                    node = lnode
                else:
                    break
            else:
                break

        min_val = self.heap.pop(-1)
        return min_val


    def parent(self, index):
        parent = (index - 1) // 2
        return parent


    def left_child(self, index):
        left_child = (2 * index) + 1
        return left_child


    def right_child(self, index):
        right_child = (2 * index) + 2
        return right_child


    def swap(self, index1, index2):
        self.heap[index1], \
        self.heap[index2] = self.heap[index2], \
                            self.heap[index1]
        pass