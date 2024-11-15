import random
from collections.abc import MutableMapping
from typing import Optional, Tuple, Generator


class Node:
    """
    Represents a node in the Cartesian Tree.

    Attributes:
        key (int): The key associated with this node.
        priority (int): The priority of the node, used to maintain heap property.
        left (Optional[Node]): Left child node.
        right (Optional[Node]): Right child node.
    """

    def __init__(self, key: int, priority: Optional[int] = None):
        self.key: int = key
        self.priority: int = priority if priority is not None else random.randint(1, 100)
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None


class CartesianTree(MutableMapping):
    """
    Implementation of a Cartesian Tree with dictionary-like behavior.

    Attributes:
        root (Optional[Node]): Root node of the Cartesian Tree.
        _size (int): Number of elements in the Cartesian Tree.
    """

    def __init__(self):
        self.root: Optional[Node] = None
        self._size: int = 0

    def _split(self, root: Optional[Node], key: int) -> Tuple[Optional[Node], Optional[Node]]:
        """
        Splits the tree rooted at `root` into two subtrees based on `key`.

        Args:
            root (Optional[Node]): The root node of the tree to be split.
            key (int): Key to split the tree on.

        Returns:
            Tuple[Optional[Node], Optional[Node]]: A tuple containing the left and right subtrees.
        """
        if not root:
            return None, None
        elif key < root.key:
            left, root.left = self._split(root.left, key)
            return left, root
        else:
            root.right, right = self._split(root.right, key)
            return root, right

    def _merge(self, left: Optional[Node], right: Optional[Node]) -> Optional[Node]:
        """
        Merges two subtrees `left` and `right` into a single tree.

        Args:
            left (Optional[Node]): Left subtree.
            right (Optional[Node]): Right subtree.

        Returns:
            Optional[Node]: The root node of the merged tree.
        """
        if not left or not right:
            return left if left else right
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            return left
        else:
            right.left = self._merge(left, right.left)
            return right

    def _insert(self, root: Optional[Node], node: Node) -> Node:
        """
        Inserts a node into the tree rooted at `root`.

        Args:
            root (Optional[Node]): The root of the tree.
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

    def _delete(self, root: Optional[Node], key: int) -> Optional[Node]:
        """
        Deletes a node with the specified `key` from the tree.

        Args:
            root (Optional[Node]): The root of the tree.
            key (int): Key of the node to delete.

        Returns:
            Optional[Node]: The new root after deletion.
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

    def __getitem__(self, key: int) -> int:
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

    def __setitem__(self, key: int, priority: Optional[int] = None) -> None:
        """
        Inserts or updates a node with the specified key and priority.

        Args:
            key (int): Key of the node.
            priority (Optional[int], optional): Priority of the node. If None, priority is auto-generated.
        """
        new_node = Node(key, priority)
        if key in self:
            self.root = self._delete(self.root, key)
        self.root = self._insert(self.root, new_node)
        self._size += 1

    def __delitem__(self, key: int) -> None:
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

    def __contains__(self, key: int) -> bool:
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

    def __iter__(self) -> Generator[int, None, None]:
        """In-order traversal iterator for the tree."""
        return self._in_order_traversal(self.root)

    def _in_order_traversal(self, node: Optional[Node]) -> Generator[int, None, None]:
        if node:
            yield from self._in_order_traversal(node.left)
            yield node.key
            yield from self._in_order_traversal(node.right)

    def __reversed__(self) -> Generator[int, None, None]:
        """Reverse-order traversal iterator for the tree."""
        return self._reverse_order_traversal(self.root)

    def _reverse_order_traversal(self, node: Optional[Node]) -> Generator[int, None, None]:
        if node:
            yield from self._reverse_order_traversal(node.right)
            yield node.key
            yield from self._reverse_order_traversal(node.left)

    def __len__(self) -> int:
        """Returns the number of elements in the tree."""
        return self._size
