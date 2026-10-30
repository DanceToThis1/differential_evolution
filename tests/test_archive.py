import numpy as np

from delab.archive import Archive


def test_archive_keeps_a_copy_of_added_vectors():
    archive = Archive(max_size=3, rng=np.random.default_rng(42))
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])

    archive.add(vectors)
    vectors[:] = 999

    # 修改原数组之后，归档仍应保存加入时的值。
    np.testing.assert_array_equal(archive.as_array(), [[1.0, 2.0], [3.0, 4.0]])


def test_archive_trim_keeps_exactly_max_size_members_when_over_capacity():
    archive = Archive(max_size=2, rng=np.random.default_rng(42))
    archive.add(np.array([[1.0], [2.0], [3.0], [4.0]]))
    assert len(archive.as_array()) == 4

    archive.trim()

    # 随机留下哪些成员不固定，容量要求是固定的。
    assert len(archive.as_array()) == archive.max_size
