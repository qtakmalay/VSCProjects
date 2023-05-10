from typing import Any, Generator, Tuple

from tree_node import TreeNode


class BinarySearchTree:
    """Binary-Search-Tree implemented for didactic reasons."""

    def __init__(self, root: TreeNode = None):
        """Initialize BinarySearchTree.

        Args:
            root (TreeNode, optional): Root of the BST. Defaults to None.
        
        Raises:
            ValueError: root is neither a TreeNode nor None.
        """
        self._root = root
        self._size = 0 if root is None else 1
        self._num_of_comparisons = 0

    def insert(self, key: int, value: Any) -> None:
        """Insert a new node into BST.

        Args:
            key (int): Key which is used for placing the value into the tree.
            value (Any): Value to insert.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is already present in the tree.
        """
        if not isinstance(key, int):
            raise ValueError("Key must be an integer.")
        if self._root is None:
            self._root = TreeNode(key, value)
        else:
            node = self._root
            while True:
                if key < node.key:
                    if node.left is None:
                        node.left = TreeNode(key, value, parent=node)
                        break
                    node = node.left
                elif key > node.key:
                    if node.right is None:
                        node.right = TreeNode(key, value, parent=node)
                        break
                    node = node.right
                else:
                    raise KeyError("Key already present in tree.")
        self._size += 1

    def find(self, key: int) -> TreeNode:
        """Return node with given key.

        Args:
            key (int): Key of node.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is not present in the tree.

        Returns:
            TreeNode: Node
        """
        if not isinstance(key, int):
            raise ValueError("Key must be an integer.")
        node = self._root
        while node is not None:
            self._num_of_comparisons += 1
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        raise KeyError("Key not present in tree.")


    @property
    def size(self) -> int:
        """Return number of nodes contained in the tree."""
        pass
        # TODO

    # If users instead call `len(tree)`, this makes it return the same as `tree.size`
    __len__ = size 

    # This is what gets called when you call e.g. `tree[5]`
    def __getitem__(self, key: int) -> Any:
        """Return value of node with given key.

        Args:
            key (int): Key to look for.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is not present in the tree.

        Returns:
            Any: [description]
        """
        return self.find(key).value

    def remove(self, key: int) -> None:
        """Remove node with given key, maintaining BST-properties.

        Args:
            key (int): Key of node which should be deleted.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is not present in the tree.
        """
        if not isinstance(key, int):
            raise ValueError("Key must be an integer")

        node_to_remove = self.find(key)
        
        if node_to_remove.left and node_to_remove.right:
            # Find the next largest node (smallest in the right subtree)
            next_node = node_to_remove.right
            while next_node.left:
                next_node = next_node.left
            
            node_to_remove.key = next_node.key
            node_to_remove.value = next_node.value
            node_to_remove = next_node

        if node_to_remove.left:
            child = node_to_remove.left
        else:
            child = node_to_remove.right

        if child:
            child.parent = node_to_remove.parent

        if node_to_remove.parent is None:
            self._root = child
        else:
            if node_to_remove.parent.left is node_to_remove:
                node_to_remove.parent.left = child
            else:
                node_to_remove.parent.right = child

        self._size -= 1
        del node_to_remove
       
    # Hint: The following 3 methods can be implemented recursively, and 
    # the keyword `yield from` might be extremely useful here:
    # http://simeonvisser.com/posts/python-3-using-yield-from-in-generators-part-1.html

    # Also, we use a small syntactic sugar here: 
    # https://www.pythoninformer.com/python-language/intermediate-python/short-circuit-evaluation/

    def inorder(self, node: TreeNode = None) -> Generator[TreeNode, None, None]:
        """Yield nodes in inorder."""
        node = node or self._root
        # This is needed in the case that there are no nodes.
        if not node:
            return iter(())
        yield from self._inorder(node)

    def preorder(self, node: TreeNode = None) -> Generator[TreeNode, None, None]:
        """Yield nodes in preorder."""
        node = node or self._root
        if not node:
            return iter(())
        yield from self._preorder(node)

    def postorder(self, node: TreeNode = None) -> Generator[TreeNode, None, None]:
        """Yield nodes in postorder."""
        node = node or self._root
        if not node:
            return iter(())
        yield from self._postorder(node)

    # this allows for e.g. `for node in tree`, or `list(tree)`.
    def __iter__(self) -> Generator[TreeNode, None, None]: 
        yield from self._preorder(self._root)

    @property
    def is_valid(self) -> bool:
        """Return if the tree fulfills BST-criteria."""
        def _is_valid_helper(node, min_val, max_val):
            if node is None:
                return True
            if not min_val <= node.key <= max_val:
                return False
            return (_is_valid_helper(node.left, min_val, node.key - 1) and
                    _is_valid_helper(node.right, node.key + 1, max_val))
        return _is_valid_helper(self._root, float("-inf"), float("inf"))

    def return_min_key(self) -> TreeNode:
        """Return the node with the smallest key (None if tree is empty)."""
        return self._inorder(self._root)[0] if self._root is not None else None
           
    def find_comparison(self, key: int) -> Tuple[int, int]:
        """Create an inbuilt python list of BST values in preorder and compute the number of comparisons needed for
           finding the key both in the list and in the BST.
           Return the numbers of comparisons for both, the list and the BST
        """
        python_list = list(node.key for node in self._preorder())
        bst_node = self.find(key)
        bst_num_of_comparisons = self._num_of_comparisons
        list_num_of_comparisons = python_list.index(key) + 1
        return (list_num_of_comparisons, bst_num_of_comparisons)

    def __repr__(self) -> str:
        return f"BinarySearchTree({list(self._inorder(self._root))})"

    ####################################################
    # Helper Functions
    ####################################################

    def get_root(self):
        return self._root

    def _inorder(self, current_node):
        if current_node is None:
            return []
        left_list = self._inorder(current_node.left)
        right_list = self._inorder(current_node.right)
        return left_list + [current_node] + right_list

    def _preorder(self, current_node):
        if current_node is None:
            return []
        left_list = self._preorder(current_node.left)
        right_list = self._preorder(current_node.right)
        return [current_node] + left_list + right_list

    def _postorder(self, current_node):
        if current_node is None:
            return []
        left_list = self._postorder(current_node.left)
        right_list = self._postorder(current_node.right)
        return left_list + right_list + [current_node]

    # You can of course add your own methods and/or functions!
    # (A method is within a class, a function outside of it.)
