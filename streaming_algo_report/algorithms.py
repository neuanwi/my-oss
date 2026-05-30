"""Streaming algorithm implementations: Bloom Filter & Count-Min Sketch.
Pure-Python, no external probabilistic-DS libraries — core logic is hand-written.
Hashing uses Python's hashlib (md5/sha) seeded by an index for independent hash funcs.
"""
import hashlib
import math


def _hashes(item, k, m, salt=b""):
    """Generate k independent hash values in [0, m) using double hashing."""
    item = str(item).encode()
    h1 = int(hashlib.md5(salt + item).hexdigest(), 16)
    h2 = int(hashlib.sha1(salt + item).hexdigest(), 16)
    return [(h1 + i * h2) % m for i in range(k)]


class BloomFilter:
    """Approximate set-membership. Bit array + k hash functions.
    Guarantees: no false negatives; false positives at rate ~ (1 - e^(-kn/m))^k.
    """
    def __init__(self, m_bits, k_hashes):
        self.m = m_bits
        self.k = k_hashes
        self.bits = bytearray((m_bits + 7) // 8)
        self.n_added = 0

    def _set(self, pos):
        self.bits[pos >> 3] |= (1 << (pos & 7))

    def _get(self, pos):
        return (self.bits[pos >> 3] >> (pos & 7)) & 1

    def add(self, item):
        for pos in _hashes(item, self.k, self.m):
            self._set(pos)
        self.n_added += 1

    def __contains__(self, item):
        return all(self._get(pos) for pos in _hashes(item, self.k, self.m))

    def size_bytes(self):
        return len(self.bits)

    @staticmethod
    def optimal_params(n, fp_rate):
        m = math.ceil(-(n * math.log(fp_rate)) / (math.log(2) ** 2))
        k = max(1, round((m / n) * math.log(2)))
        return m, k


class CountMinSketch:
    """Approximate frequency estimation. depth x width counter matrix.
    Guarantees: estimate >= true count (overestimate only);
    error <= eps*N with prob 1-delta where width=ceil(e/eps), depth=ceil(ln(1/delta)).
    """
    def __init__(self, width, depth):
        self.w = width
        self.d = depth
        self.table = [[0] * width for _ in range(depth)]

    def _positions(self, item):
        # one hash per row, salted by row index
        return [_hashes(item, 1, self.w, salt=bytes([r]))[0] for r in range(self.d)]

    def add(self, item, count=1):
        for r, pos in enumerate(self._positions(item)):
            self.table[r][pos] += count

    def estimate(self, item):
        return min(self.table[r][pos] for r, pos in enumerate(self._positions(item)))

    def size_bytes(self):
        # 8 bytes per integer counter (CPython int approx; conceptual size)
        return self.w * self.d * 8

    @staticmethod
    def optimal_params(eps, delta):
        width = math.ceil(math.e / eps)
        depth = math.ceil(math.log(1 / delta))
        return width, depth
