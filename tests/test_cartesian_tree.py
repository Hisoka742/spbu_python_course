import pytest
from project.Tree.CartesianTree import CartesianTree


def test_cartesian_tree_insertion_and_retrieval():
    tree = CartesianTree()
    tree[10] = 30  # Insert key=10 with priority=30
    tree[20] = 40
    tree[5] = 25

    assert tree[10] == 30
    assert tree[20] == 40
    assert tree[5] == 25


def test_cartesian_tree_overwrite():
    tree = CartesianTree()
    tree[10] = 30
    tree[10] = 50  # Overwrite priority

    assert tree[10] == 50


def test_cartesian_tree_deletion():
    tree = CartesianTree()
    tree[10] = 30
    tree[20] = 40
    del tree[10]

    with pytest.raises(KeyError):
        _ = tree[10]  # Should raise KeyError since 10 was deleted
    assert 20 in tree  # Key 20 should still exist


def test_cartesian_tree_contains():
    tree = CartesianTree()
    tree[15] = 45

    assert 15 in tree
    assert 10 not in tree


def test_cartesian_tree_iteration():
    tree = CartesianTree()
    tree[10] = 30
    tree[20] = 40
    tree[5] = 25

    # In-order traversal should yield sorted keys
    keys = list(iter(tree))
    assert keys == [5, 10, 20]


def test_cartesian_tree_reverse_iteration():
    tree = CartesianTree()
    tree[10] = 30
    tree[20] = 40
    tree[5] = 25

    # Reverse-order traversal should yield keys in descending order
    keys = list(reversed(tree))
    assert keys == [20, 10, 5]


def test_cartesian_tree_len():
    tree = CartesianTree()
    tree[10] = 30
    tree[20] = 40
    tree[5] = 25

    assert len(tree) == 3

    del tree[10]
    assert len(tree) == 2


def test_cartesian_tree_keyerror_on_missing_key():
    tree = CartesianTree()
    with pytest.raises(KeyError):
        _ = tree[99]  # Accessing a missing key should raise KeyError


def test_cartesian_tree_priority_insertion():
    tree = CartesianTree()
    tree[1] = 100
    tree[2] = 50  # Should not replace 1 due to lower priority unless 1 is removed

    assert tree[1] == 100
    assert tree[2] == 50


if __name__ == "__main__":
    pytest.main()
