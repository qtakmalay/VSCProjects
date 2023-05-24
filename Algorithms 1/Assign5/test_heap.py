class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def insertKey(self, k):
        heappos = len(self.heap)
        self.heap.append(k)

        while(heappos != 0 and self.heap[self.parent(heappos)] > self.heap[heappos]):
            self.heap[heappos], self.heap[self.parent(heappos)] = self.heap[self.parent(heappos)], self.heap[heappos]
            heappos = self.parent(heappos)

    def printHeap(self):
        print(self.heap)

heap = MinHeap()
numbers = [107, 79, 121, 59, 48, 62, 23, 47, 22, 19, 24, 2, 6]

for num in numbers:
    heap.insertKey(num)
    heap.printHeap()
