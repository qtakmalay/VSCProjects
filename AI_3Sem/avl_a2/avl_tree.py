from avl_node import AVLNode


class AVLTree:
    class NodeGroup:
        def __init__(self):
            self.a = None
            self.b = None
            self.c = None
            self.t0 = None
            self.t1 = None
            self.t2 = None
            self.t3 = None

    def __init__(self):
        self.root = None
        self.size = 0
        self.to_restruct = None

    def get_tree_root(self):
        """
        Method to get the root node of the AVLTree
        :return AVLNode -- the root node of the AVL tree
        """
        # TODO
        return self.root

    def get_tree_height(self):
        """Retrieves tree height.
        :return -1 in case of empty tree, current tree height otherwise.
        """
        # TODO
        return self._get_height(self.root)

    def get_tree_size(self):
        """Return number of key/value pairs in the tree.
        :return Number of key/value pairs.
        """
        # TODO
        return self.size

    def find_by_key(self, key):
        """Returns value of node with given key.
        :param key: Key to search.
        :return Corresponding value if key was found, None otherwise.
        :raises ValueError if the key is None
        """
        if key is None:
            raise ValueError("Cannot search for null key!")
        current = self.root
        while current is not None:
            if current.key == key:
                return current.value
            elif current.key < key:
                current = current.right
            else:
                current = current.left

        return None


    def insert(self, key, value):
        """Inserts a new node into AVL tree.
        :param key: Key of the new node.
        :param value: Data of the new node. Must not be None. Nodes with the same key
        are not allowed. In this case False is returned. None-Keys and None-Values are
        not allowed. In this case an error is raised.
        :return True if the insert was successful, False otherwise.
        :raises ValueError if the key or value is None.
        """
        if key is None:
            raise ValueError("Null keys are not allowed!")

        if self.root is None:
            self.root = AVLNode(key, value)
            inserted_node = self.root
        else:
            current = self.root
            while True:
                if current.key == key:
                    return False
                elif current.key < key:
                    if current.right is not None:
                        current = current.right
                    else:
                        new_node = AVLNode(key, value)
                        self.set_right(current, new_node)
                        inserted_node = new_node
                        break
                else:
                    if current.left is not None:
                        current = current.left
                    else:
                        new_node = AVLNode(key, value)
                        self.set_left(current, new_node)
                        inserted_node = new_node
                        break
        self.size += 1

        self._update_heights_and_rebalance(inserted_node)

        return True

    def remove_by_key(self, key):
        """Removes node with given key.
        :param key: Key of node to remove.
        :return True If node was found and deleted, False otherwise.
        @raises ValueError if the key is None.
        """
        if key is None:
            raise ValueError("Null key is not allowed!")

        parent = None
        current = self.root
        new_sub_root = None

        while not (current is None):
            if current.key == key:
                if parent is None:
                    self.root = self._remove_bst(current)
                    if self.root is not None:
                        self.root.parent = None
                elif parent.left == current:
                    new_sub_root = self._remove_bst(current)
                    self.set_left(parent, new_sub_root)
                elif parent.right == current:
                    new_sub_root = self._remove_bst(current)
                    self.set_right(parent, new_sub_root)
                else:
                    raise ValueError()

                self.size -= 1
                # to_restruct is the node from which the search for the first unbalanced node is started
                if self.to_restruct is not None:
                    current = self.to_restruct
                    while current is not None:
                        current.height = 1 + max(self._get_height(current.left), self._get_height(current.right))
                        balance = self._get_balance(current)

                        if balance > 1 and self._get_balance(current.left) >= 0:
                            self._right_rotate(current)

                        elif balance > 1 and self._get_balance(current.left) < 0:
                            current.left = self._left_rotate(current.left)
                            self._right_rotate(current)

                        elif balance < -1 and self._get_balance(current.right) <= 0:
                            self._left_rotate(current)

                        elif balance < -1 and self._get_balance(current.right) > 0:
                            current.right = self._right_rotate(current.right)
                            self._left_rotate(current)

                        current = current.parent
                return True
            else:
                parent = current
                if current.key > key:
                    current = current.left
                else:
                    current = current.right

        return False
#auxiliary functions start
    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

    def _right_rotate(self, y):
        x = y.left
        y.left = x.right
        if x.right is not None:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x

        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))

    def _get_height(self, node):
        if node is None: return -1
        return node.height

    def _get_balance(self, node):
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    
    def _update_heights_and_rebalance(self, node):
        current = node
        while current is not None:
            current.height = 1 + max(self._get_height(current.left), self._get_height(current.right))
            balance = self._get_balance(current)
            if balance > 1 and self._get_balance(current.left) >= 0:
                self._right_rotate(current)
            elif balance > 1 and self._get_balance(current.left) < 0:
                self._left_rotate(current.left)
                self._right_rotate(current)
            elif balance < -1 and self._get_balance(current.right) <= 0:
                self._left_rotate(current)
            elif balance < -1 and self._get_balance(current.right) > 0:
                self._right_rotate(current.right)
                self._left_rotate(current)

            current = current.parent
#auxiliary functions end

    def _remove_bst(self, old_sub_root):
        new_sub_root = None
        if old_sub_root.left is None and old_sub_root.right is None:
            new_sub_root = None
            self.to_restruct = old_sub_root.parent
        elif old_sub_root.left is None:
            new_sub_root = old_sub_root.right
            self.to_restruct = new_sub_root
        elif old_sub_root.right is None:
            new_sub_root = old_sub_root.left
            self.to_restruct = new_sub_root
        elif old_sub_root.left.right is None:
            new_sub_root = old_sub_root.left
            self.set_right(new_sub_root, old_sub_root.right)
            self.to_restruct = new_sub_root
        elif old_sub_root.right.left is None:
            new_sub_root = old_sub_root.right
            self.set_left(new_sub_root, old_sub_root.left)
            self.to_restruct = new_sub_root
        else:
            new_sub_root = old_sub_root.left
            while new_sub_root.right is not None:
                new_sub_root = new_sub_root.right
            predecessor_p = new_sub_root.parent
            self.set_right(predecessor_p, new_sub_root.left)
            self.set_right(new_sub_root, old_sub_root.right)
            self.set_left(new_sub_root, old_sub_root.left)
            self.to_restruct = predecessor_p

        return new_sub_root

    def set_left(self, parent, child):
        parent.left = child
        if child is not None:
            child.parent = parent

    def set_right(self, parent, child):
        parent.right = child
        if child is not None:
            child.parent = parent

