class MaxHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def leftChild(self, i):
        return 2 * i + 1

    def rightChild(self, i):
        return 2 * i + 2

    def insertKey(self, k):
        heapsize = len(self.heap)
        self.heap.append(k)
        
        while heapsize != 0 and self.heap[self.parent(heapsize)] < self.heap[heapsize]:
            self.heap[heapsize], self.heap[self.parent(heapsize)] = self.heap[self.parent(heapsize)], self.heap[heapsize]
            heapsize = self.parent(heapsize)

    def removeMax(self):
        if len(self.heap) == 0:
            return None
        
        root = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap = self.heap[:-1]
        self.heapify(0)

        return root

    def heapify(self, i):
        largest = i
        left = self.leftChild(i)
        right = self.rightChild(i)

        if left < len(self.heap) and self.heap[largest] < self.heap[left]:
            largest = left
        if right < len(self.heap) and self.heap[largest] < self.heap[right]:
            largest = right
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self.heapify(largest)

    def printHeap(self):
        print(self.heap)


heap = MaxHeap()
numbers = [54,	28,	39,	8,	17,	20,	21,	5,	2,	15,	1]

for num in numbers:
    heap.insertKey(num)
    heap.printHeap()

heap.removeMax()
heap.printHeap()
print("""54	28	39	8	17	20	21	5	2	15	1	The initial MaxHeap
1	28	39	8	17	20	21	5	2	15	54	Swap last element with root and remove the last element (54)
39	28	1	8	17	20	21	5	2	15		Swap 1 with larger child (39)
39	28	20	8	17	1	21	5	2	15		Swap 1 with larger child (20)
39	28	21	8	17	1	20	5	2	15		Swap 20 with larger child (21)
39	28	21	8	17	20	1	5	2	15		Swap 1 with larger child (20)
39	28	21	8	17	20	15	5	2	1		Move 1 to last position, MaxHeap is complete""".replace("	", "           "))