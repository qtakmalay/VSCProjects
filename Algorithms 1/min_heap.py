from typing import Optional


class MinHeap:
    def __init__(self):
        self.heap = []
        self.size = len(self.heap)
        self.node = None
    def get_size(self) -> int:
        """
        @return number of elements in the min heap
        """
        return self.size

    def is_empty(self) -> bool:
        """
        @return True if the min heap is empty, False otherwise
        """
        return len(self.heap) == 0

    def insert(self, integer_val: int) -> None:
        """
        inserts integer_val into the min heap
        @param integer_val: the value to be inserted
        @raises ValueError if integer_val is None or not an int
        """
        if not isinstance(integer_val, int) or integer_val is None:
            raise ValueError("integer_val is None or not an int")
        self.heap.append(integer_val)
        self.size += 1
        self.up_heap(len(self.heap) - 1)


    def get_min(self) -> Optional[int]:
        """
        returns the value of the minimum element of the PQ without removing it
        @return the minimum value of the PQ or None if no element exists
        """
        if self.is_empty(): 
            return 
        else:
            return self.heap[0]



    def remove_min(self) -> Optional[int]:
        """
        removes the minimum element from the PQ and returns its value
        @return the value of the removed element or None if no element exists
        """
        if self.is_empty():
            return
        min_val = self.get_min()
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self.down_heap(0)
        self.size -= 1
        return min_val
            
    def up_heap(self, index):
        p_idx = self.parent(index)
        if index > 0 and self.heap[p_idx] > self.heap[index]:
            self.swap(p_idx, index)
            self.up_heap(p_idx)

    def down_heap(self, index):
        min_idx = index
        l_child_idx = self.left_child(index)
        r_child_idx = self.right_child(index)
        if l_child_idx < len(self.heap) and self.heap[l_child_idx] < self.heap[min_idx]:
            min_idx = l_child_idx
        if r_child_idx < len(self.heap) and self.heap[r_child_idx] < self.heap[min_idx]:
            min_idx = r_child_idx
        if min_idx != index:
            self.swap(index, min_idx)
            self.down_heap(min_idx)

    def parent(self, index):
        return (index - 1) // 2

    def left_child(self, index):
        return (2 * index) + 1

    def right_child(self, index):
        return (2 * index) + 2

    def swap(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]