import random
from collections.abc import MutableMapping


class Node:
    def __init__(self, key, priority=None):
        self.key = key
        self.priority = priority if priority is not None else random.randint(1, 100)
        self.left = None
        self.right = None


class CartesianTree(MutableMapping):
    def __init__(self):
        self.root = None
        self._size = 0

    def _split(self, root, key):
        if not root:
            return None, None
        elif key < root.key:
            left, root.left = self._split(root.left, key)
            return left, root
        else:
            root.right, right = self._split(root.right, key)
            return root, right

    def _merge(self, left, right):
        if not left or not right:
            return left if left else right
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            return left
        else:
            right.left = self._merge(left, right.left)
            return right

    def _insert(self, root, node):
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
        new_node = Node(key, priority)
        if key in self:
            self.root = self._delete(self.root, key)
        self.root = self._insert(self.root, new_node)
        self._size += 1

    def __delitem__(self, key):
        if key not in self:
            raise KeyError("Key not found")
        self.root = self._delete(self.root, key)
        self._size -= 1

    def __contains__(self, key):
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
        return self._in_order_traversal(self.root)

    def _in_order_traversal(self, node):
        if node:
            yield from self._in_order_traversal(node.left)
            yield node.key
            yield from self._in_order_traversal(node.right)

    def __reversed__(self):
        return self._reverse_order_traversal(self.root)

    def _reverse_order_traversal(self, node):
        if node:
            yield from self._reverse_order_traversal(node.right)
            yield node.key
            yield from self._reverse_order_traversal(node.left)

    def __len__(self):
        return self._size
