from my_list_node import MyListNode


class MySortedDoublyLinkedList:
    """A base class providing a doubly linked list representation."""

    def __init__(self, head: 'MyListNode' = None, tail: 'MyListNode' = None, size: int = 0) -> None:
        """Create a list and default values are None."""
        self._head = head
        self._tail = tail
        self._size = size

    def __len__(self) -> int:
        """Return the number of elements in the list."""
        return self._size

    def __str__(self) -> str:
        """Return linked list in string representation."""
        result = []
        node = self._head
        while node:
            result.append(node.elem)
            node = node.next_node
        return str(result)

    # The following methods have to be implemented.

    def get_value(self, index: int) -> int:
        """Return the value (elem) at position 'index' without removing the node.

        Args:
            index (int): 0 <= index < length of list

        Returns:
            (int): Retrieved value.

        Raises:
            ValueError: If the passed index is not an int or out of range.
        """
        try:
            if not isinstance(index,int) or index < 0 or index > self._size-1:
                raise ValueError

            node = self._head
            for i in range(self._size):
                if(i == index):
                    return node.elem
                else:
                    node = node.next_node
                    
        except ValueError as ex:
            raise ex

    def search_value(self, val: int) -> int:
        """Return the index of the first occurrence of 'val' in the list.

        Args:
            val (int): Value to be searched.

        Returns:
            (int): Retrieved index.

        Raises:
            ValueError: If val is not an int.
        """
        try:
            if not isinstance(val,int):
                raise ValueError

            node = self._head
            for i in range(self._size):
                if(val == node.elem):
                    return i
                else:
                    node = node.next_node
            return -1
        except ValueError as ex:
            raise ex

    def insert(self, val: int) -> None:
        """Add a new node containing 'val' to the list, keeping the list in ascending order.

        Args:
            val (int): Value to be added.

        Raises:
            ValueError: If val is not an int.
        """
        # new_node = MyListNode(val)
        # if self.size == 0:
        #     self._head = new_node
        #     self._tail = new_node
        # if val < self._head.elem:
        #     new_node.next_node = self._head
        #     self._head = new_node
        #     self._head.next_node =  
        if not isinstance(val, int):
            raise ValueError("Value should be an integer.")

        new_node = MyListNode(val)

        if self._size == 0:
            self._head = self._tail = new_node
        elif val < self._head.elem:
            new_node.next_node = self._head
            self._head.prev_node = new_node
            self._head = new_node
        elif val > self._tail.elem:
            new_node.prev_node = self._tail
            self._tail.next_node = new_node
            self._tail = new_node
        else:
            current = self._head
            while current.next_node and current.next_node.elem <= val:
                current = current.next_node
            new_node.prev_node = current
            new_node.next_node = current.next_node
            if current.next_node:
                current.next_node.prev_node = new_node
            current.next_node = new_node

        self._size += 1
            

        
            

    def remove_first(self, val: int) -> bool:
        """Remove the first occurrence of the parameter 'val'.

        Args:
            val (int): Value to be removed.

        Returns:
            (bool): Whether an element was successfully removed or not.

        Raises:
            ValueError: If val is not an int.
        """
        try:
            if not isinstance(val,int):
                raise ValueError
            cur = self._head
            if self._size == 0:
                return False
            if cur.elem == val and val == self._head.elem:
                self._head = cur.next_node
                self._size -= 1
                return True
            while cur:
                if cur.elem == val and val == self._tail.elem:
                    temp = cur
                    cur.prev_node.next_node = None
                    temp.prev_node = cur.prev_node
                    cur.prev_node = None
                    cur.next_node = None
                    cur.elem = None
                    self._size -= 1
                    return True
                if cur.elem == val:
                    temp = cur.next_node
                    cur.prev_node.next_node = temp
                    temp.prev_node = cur.prev_node
                    self._size -= 1
                    return True
                else:
                    cur = cur.next_node
            return False
        except ValueError as ex:
            raise ex

    def remove_all(self, val: int) -> bool:
        """Remove all occurrences of the parameter 'val'.

        Args:
            val (int): Value to be removed.

        Returns:
            (bool): Whether elements were successfully removed or not.

        Raises:
            ValueError: If val is not an int.
        """
        try:
            if not isinstance(val,int):
                raise ValueError
            if self._size == 0:
                return False
            cur = self._head
            rem_status = False
            while cur:
                if val == cur.elem:
                    rem_status = True
                    if cur.prev_node:
                        cur.prev_node.next_node = cur.next_node
                    else:
                        self._head = cur.next_node
                    if cur.next_node:
                        cur.next_node.prev_node = cur.prev_node
                    else:
                        self._tail = cur.prev_node
                    self._size -= 1
                cur = cur.next_node
            return rem_status

        except ValueError as ex:
            raise ex

    def remove_duplicates(self) -> None:
        """Remove all duplicate occurrences of values from the list."""
        # TODO

    def filter_n_max(self, n: int) -> None:
        """Filter the list to only contain the 'n' highest values.

        Args:
            n (int): 0 < n <= length of list

        Raises:
            ValueError: If the passed value n is not an int or out of range.
        """
        # TODO

    def filter_odd(self) -> None:
        """Filter the list to only contain odd values."""
        # TODO
