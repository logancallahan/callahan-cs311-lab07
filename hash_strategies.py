"""
Lab 7: The Collision Resolver -- starter.

Complete the three classes below. See
Lab_07_The_Collision_Resolver.md, Part B, for the full requirements.
"""

from typing import Generic, Hashable, List, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_TOMBSTONE = object()  # sentinel marking a deleted open-addressing slot


class _ChainNode(Generic[K, V]):
    __slots__ = ("key", "value", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.next: Optional["_ChainNode[K, V]"] = None


class ChainedHashMap(Generic[K, V]):
    """Separate chaining: each bucket is a linked list of (key, value)."""

    def __init__(self, initial_size: int = 16) -> None:
        self._buckets: List[Optional[_ChainNode[K, V]]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Insert, or update in place if `key` already exists. Resize (double + rehash) once load factor > 0.75."""
        buckets = self._buckets
        index = hash(key) % len(buckets)
        node = buckets[index]
        while node is not None:
            if node.key == key:
                node.value = value
                return
            node = node.next

        node = _ChainNode(key, value)
        node.next = buckets[index]
        buckets[index] = node
        self._count += 1

        if self._count * 4 > len(buckets) * 3:
            new_size = len(buckets) * 2
            new_buckets: List[Optional[_ChainNode[K, V]]] = [None] * new_size
            for node in buckets:
                while node is not None:
                    next_node = node.next
                    index = hash(node.key) % new_size
                    node.next = new_buckets[index]
                    new_buckets[index] = node
                    node = next_node
            self._buckets = new_buckets

    def get(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        node = self._buckets[hash(key) % len(self._buckets)]
        while node is not None:
            if node.key == key:
                return node.value
            node = node.next
        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key`. Raise KeyError if missing."""
        index = hash(key) % len(self._buckets)
        node = self._buckets[index]
        previous = None
        while node is not None:
            if node.key == key:
                if previous is None:
                    self._buckets[index] = node.next
                else:
                    previous.next = node.next
                self._count -= 1
                return
            previous = node
            node = node.next
        raise KeyError(key)


class LinearProbingHashMap(Generic[K, V]):
    """Open addressing with linear probing and tombstone deletion."""

    def __init__(self, initial_size: int = 16) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Resize (double + rehash) once load factor > 0.7."""
        keys = self._keys
        size = len(keys)
        index = hash(key) % size
        first_deleted = -1
        for _ in range(size):
            stored_key = keys[index]
            if stored_key is None:
                break
            if stored_key is _TOMBSTONE:
                if first_deleted == -1:
                    first_deleted = index
            elif stored_key == key:
                self._values[index] = value
                return
            index = (index + 1) % size

        if first_deleted != -1:
            index = first_deleted
        keys[index] = key
        self._values[index] = value
        self._count += 1

        if self._count * 10 > size * 7:
            new_size = size * 2
            new_keys: List[object] = [None] * new_size
            new_values: List[Optional[V]] = [None] * new_size
            for old_index, stored_key in enumerate(keys):
                if stored_key is None or stored_key is _TOMBSTONE:
                    continue
                index = hash(stored_key) % new_size
                while new_keys[index] is not None:
                    index = (index + 1) % new_size
                new_keys[index] = stored_key
                new_values[index] = self._values[old_index]
            self._keys = new_keys
            self._values = new_values

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        keys = self._keys
        size = len(keys)
        index = hash(key) % size
        for _ in range(size):
            stored_key = keys[index]
            if stored_key is None:
                break
            if stored_key is not _TOMBSTONE and stored_key == key:
                return self._values[index]
            index = (index + 1) % size
        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone (not None) so later probes don't stop early. Raise KeyError if missing."""
        keys = self._keys
        size = len(keys)
        index = hash(key) % size
        for _ in range(size):
            stored_key = keys[index]
            if stored_key is None:
                break
            if stored_key is not _TOMBSTONE and stored_key == key:
                keys[index] = _TOMBSTONE
                self._values[index] = None
                self._count -= 1
                return
            index = (index + 1) % size
        raise KeyError(key)


class QuadraticProbingHashMap(Generic[K, V]):
    """
    Open addressing with quadratic probing and tombstone deletion.

    Pitfall to design around: with a power-of-2 table size, the probe
    sequence (idx + i^2) mod size does NOT reach every slot -- it can
    cycle through only about half of them, so the table can appear
    "full" and raise/loop forever even though empty slots exist
    elsewhere. Two standard fixes, pick one:
      (a) use a PRIME table size (so the quadratic sequence covers all
          slots whenever load factor < 1), or
      (b) resize proactively -- check load factor BEFORE attempting an
          insert's probe sequence, not only after a successful insert.
    Using both is safest.
    """

    def __init__(self, initial_size: int = 17) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Resize (grow + rehash) once load factor > 0.7 -- see the pitfall note above."""
        key_hash = hash(key)
        while True:
            keys = self._keys
            size = len(keys)
            first_available = -1
            for step in range(size):
                index = (key_hash + step * step) % size
                stored_key = keys[index]
                if stored_key is None:
                    if first_available == -1:
                        first_available = index
                    break
                if stored_key is _TOMBSTONE:
                    if first_available == -1:
                        first_available = index
                elif stored_key == key:
                    self._values[index] = value
                    return

            if first_available != -1 and (self._count + 1) * 10 <= size * 7:
                keys[first_available] = key
                self._values[first_available] = value
                self._count += 1
                return

            new_size = max(3, size * 2 + 1)
            while True:
                divisor = 3
                while divisor * divisor <= new_size and new_size % divisor:
                    divisor += 2
                if divisor * divisor > new_size:
                    break
                new_size += 2

            new_keys: List[object] = [None] * new_size
            new_values: List[Optional[V]] = [None] * new_size
            for old_index, stored_key in enumerate(keys):
                if stored_key is None or stored_key is _TOMBSTONE:
                    continue
                stored_hash = hash(stored_key)
                step = 0
                index = stored_hash % new_size
                while new_keys[index] is not None:
                    step += 1
                    index = (stored_hash + step * step) % new_size
                new_keys[index] = stored_key
                new_values[index] = self._values[old_index]
            self._keys = new_keys
            self._values = new_values

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        keys = self._keys
        size = len(keys)
        key_hash = hash(key)
        for step in range(size):
            index = (key_hash + step * step) % size
            stored_key = keys[index]
            if stored_key is None:
                break
            if stored_key is not _TOMBSTONE and stored_key == key:
                return self._values[index]
        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone. Raise KeyError if missing."""
        keys = self._keys
        size = len(keys)
        key_hash = hash(key)
        for step in range(size):
            index = (key_hash + step * step) % size
            stored_key = keys[index]
            if stored_key is None:
                break
            if stored_key is not _TOMBSTONE and stored_key == key:
                keys[index] = _TOMBSTONE
                self._values[index] = None
                self._count -= 1
                return
        raise KeyError(key)
