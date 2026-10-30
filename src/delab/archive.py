import numpy as np

class Archive:
    """归档个体的简单实现。"""
    def __init__(self, max_size, rng):
        self.max_size = max_size
        self.members = np.empty((0, 0), dtype=float)
        self.rng = rng

    def add(self, vectors):
        """添加新成员，可能会丢弃旧成员以保持最大容量。"""
        if self.members.size == 0:
            self.members = vectors.copy()
        else:
            self.members = np.vstack((self.members, vectors))

    def trim(self):
        """如果归档成员数超过最大容量，则随机丢弃一些成员。"""
        if len(self.members) > self.max_size:
            indices_to_keep = self.rng.choice(
                len(self.members), size=self.max_size, replace=False
            )
            self.members = self.members[indices_to_keep]

    def as_array(self):
        """返回归档成员的数组视图。"""
        return self.members