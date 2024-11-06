import random
from collections.abc import MutableMapping


class Node:
    """
    Represents a node in the Cartesian Tree.

    Attributes:
        key (int): The key associated with this node.
        priority (int): The priority of the node, used to maintain heap property.
        left (Node): Left child node.
        right (Node): Right child node.
    """

    def __init__(self, key, priority=None):
        self.key = key
        self.priority = priority if priority is not None else random.randint(1, 100)
        self.left = None
        self.right = None


class CartesianTree(MutableMapping):
    """
    Implementation of a Cartesian Tree with dictionary-like behavior.

    Attributes:
        root (Node): Root node of the Cartesian Tree.
        _size (int): Number of elements in the Cartesian Tree.
    """

    def __init__(self):
        self.root = None
        self._size = 0

    def _split(self, root, key):
        """
        Splits the tree rooted at `root` into two subtrees based on `key`.

        Args:
            root (Node): The root node of the tree to be split.
            key (int): Key to split the tree on.

        Returns:
            tuple: A tuple containing the left and right subtrees.
        """
        if not root:
            return None, None
        elif key < root.key:
            left, root.left = self._split(root.left, key)
            return left, root
        else:
            root.right, right = self._split(root.right, key)
            return root, right

    def _merge(self, left, right):
        """
        Merges two subtrees `left` and `right` into a single tree.

        Args:
            left (Node): Left subtree.
            right (Node): Right subtree.

        Returns:
            Node: The root node of the merged tree.
        """
        if not left or not right:
            return left if left else right
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            return left
        else:
            right.left = self._merge(left, right.left)
            return right

    def _insert(self, root, node):
        """
        Inserts a node into the tree rooted at `root`.

        Args:
            root (Node): The root of the tree.
            node (Node): The node to insert.

        Returns:
            Node: The new root after insertion.
        """
        if not root:
            return node
        if node.priority > root.priority:
            left, right = self._split(root, node.key)
            node.left, node.right = left, right
            return node
        elif node.key < root.key:
            root.left = self._insert(root.left, node)
        else:
            root.right = self._insert(root.right, node)
        return root

    def _delete(self, root, key):
        """
        Deletes a node with the specified `key` from the tree.

        Args:
            root (Node): The root of the tree.
            key (int): Key of the node to delete.

        Returns:
            Node: The new root after deletion.
        """
        if not root:
            return None
        if key == root.key:
            return self._merge(root.left, root.right)
        elif key < root.key:
            root.left = self._delete(root.left, key)
        else:
            root.right = self._delete(root.right, key)
        return root

    def __getitem__(self, key):
        """
        Retrieves the priority of a node by its key.

        Args:
            key (int): Key of the node.

        Returns:
            int: Priority of the node.

        Raises:
            KeyError: If the key is not found.
        """
        node = self.root
        while node:
            if key == node.key:
                return node.priority
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        raise KeyError("Key not found")

    def __setitem__(self, key, priority=None):
        """
        Inserts or updates a node with the specified key and priority.

        Args:
            key (int): Key of the node.
            priority (int, optional): Priority of the node. If None, priority is auto-generated.
        """
        new_node = Node(key, priority)
        if key in self:
            self.root = self._delete(self.root, key)
        self.root = self._insert(self.root, new_node)
        self._size += 1

    def __delitem__(self, key):
        """
        Deletes a node with the specified key.

        Args:
            key (int): Key of the node to delete.

        Raises:
            KeyError: If the key is not found.
        """
        if key not in self:
            raise KeyError("Key not found")
        self.root = self._delete(self.root, key)
        self._size -= 1

    def __contains__(self, key):
        """
        Checks if a node with the specified key exists in the tree.

        Args:
            key (int): Key of the node.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        node = self.root
        while node:
            if key == node.key:
                return True
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return False

    def __iter__(self):
        """In-order traversal iterator for the tree."""
        return self._in_order_traversal(self.root)

    def _in_order_traversal(self, node):
        if node:
            yield from self._in_order_traversal(node.left)
            yield node.key
            yield from self._in_order_traversal(node.right)

    def __reversed__(self):
        """Reverse-order traversal iterator for the tree."""
        return self._reverse_order_traversal(self.root)

    def _reverse_order_traversal(self, node):
        if node:
            yield from self._reverse_order_traversal(node.right)
            yield node.key
            yield from self._reverse_order_traversal(node.left)

    def __len__(self):
        """Returns the number of elements in the tree."""
        return self._size