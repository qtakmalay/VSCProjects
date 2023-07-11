"""
Created on May 05, 2020
Changed on June 06, 2021
Changed on May 10, 2023 by Florian Beck

@author: martin
"""
import unittest
from datetime import date

from min_heap import MinHeap

heap012 = [0, 1, 2]
heap12 = [1, 2]
heap102 = [1, 0, 2]
heap102result = [0, 1, 2]
heap210 = [0, 1, 2]
heap43210 = [0, 1, 2, 3, 4]
heap43210result = [0, 1, 2, 3, 4]
heap43210removeMinResult = [1, 3, 2, 4]
heap0 = [0]

maxPoints = 12.0  # defines the maximum achievable points for the example tested here
points = maxPoints  # stores the actually achieved points based on failed unit tests
summary = ""


def deduct_pts(value):
    global points
    points = points - value
    if points < 0:
        points = 0


def resolve_amount_of_pts_to_deduct(argument):
    pool = {
        "test_heap_insert_with_one_element": 0.5,
        "test_heap_insert_invalid": 0.25,
        "test_heap_insert_none": 0.25,
        "test_heap_insert_with_upheap_of_middle_element": 1,
        "test_heap_insert_with_upheap_of_last_element": 1,
        "test_heap_insert_without_upheap": 1,
        "test_heap_insert_with_multiple_upheap": 1,
        "test_heap_is_empty": 0.5,
        "test_heap_get_min": 1,
        "test_heap_get_min_with_multiple_elements": 1,
        "test_heap_get_min_empty_heap": 0.5,
        "test_heap_remove_min_last_element": 1,
        "test_heap_remove_min_without_downheap": 1,
        "test_heap_remove_min_with_downheap": 1,
        "test_heap_size_with_insert_and_remove": 1,
    }

    # resolve the pts to deduct from pool
    return pool.get(argument, 0)


class UnitTestTemplate(unittest.TestCase):
    def setUp(self):
        pass

    ####################################################
    # Definition of test cases
    ####################################################

    def test_heap_insert_with_one_element(self):
        heap = MinHeap()
        for i in heap0:
            heap.insert(i)
        self.assertEqual(heap0, heap.heap, f"insert of one element failed, was: {heap.heap} but should be: {heap0}")

    def test_heap_insert_with_upheap_of_middle_element(self):
        heap = MinHeap()
        for i in heap102:
            heap.insert(i)
        self.assertEqual(heap102result, heap.heap, f"insert failed, was: {heap.heap} but should be: {heap102result}")

    def test_heap_insert_with_upheap_of_last_element(self):
        heap = MinHeap()
        heap.insert(2)
        heap.insert(3)
        heap.insert(1)

        self.assertEqual(1, heap.heap[0], f"ERROR heap.insert() failed because of wrong element at index 0 was: "
                                          f"{heap.heap[0]}, should be 1 for input sequence [2,3,1]")
        self.assertEqual(3, heap.heap[1], "ERROR heap.insert() failed because of wrong element at index 1 was: "
                                          f"{heap.heap[1]}, should be 3 for input sequence [2,3,1]")
        self.assertEqual(2, heap.heap[2], "ERROR heap.insert() failed because of wrong element at index 2 was: "
                                          f"{heap.heap[2]}, should be 2 for input sequence [2,3,1]")

    def test_heap_insert_without_upheap(self):
        heap = MinHeap()
        for i in heap012:
            heap.insert(i)

        self.assertEqual(heap012, heap.heap, f"insert failed, was: {heap.heap} but should be: {heap012}")

    def test_heap_insert_with_multiple_upheap(self):
        heap = MinHeap()
        for i in heap43210:
            heap.insert(i)

        self.assertEqual(heap43210result, heap.heap, f"insert failed, was: {heap.heap} but should be: {heap43210result}")

    def test_heap_insert_invalid(self):
        heap = MinHeap()
        with self.assertRaises(ValueError, msg="ERROR: heap.insert() did not raise ValueError when trying to insert str"):
            heap.insert('test')

    def test_heap_insert_none(self):
        heap = MinHeap()
        with self.assertRaises(ValueError, msg="ERROR: heap.insert() did not raise ValueError when trying to insert None"):
            heap.insert(None)

    def test_heap_is_empty(self):
        heap = MinHeap()
        self.assertTrue(heap.is_empty(), "ERROR: heap.is_empty() -> returned False. The heap should have been empty (return True), but it returned False.")

        heap.insert(1)
        self.assertFalse(heap.is_empty(), "ERROR: heap.is_empty() -> The heap should not have been empty, but it is.")

    def test_heap_get_min(self):
        heap = MinHeap()
        heap.insert(10)
        self.assertEqual([10], heap.heap, f"get_min cannot be tested because insert created an invalid heap: "
                                          f"{heap.heap} but should be: [10]")
        self.assertEqual(10, heap.get_min(), f"ERROR: heap.get_min() returned incorrect value ({heap.get_min()}) for "
                                             f"insert sequence [10], should be 10")

    def test_heap_get_min_with_multiple_elements(self):
        heap = MinHeap()
        for i in heap43210:
            heap.insert(i)
        self.assertEqual(heap43210result, heap.heap, f"get_min cannot be tested because insert created an invalid "
                                                     f"heap: {heap.heap} but should be: {heap43210result}")
        self.assertEqual(0, heap.get_min(), f"ERROR: heap.get_min() returned incorrect value ({heap.get_min()}) for "
                                            f"insert sequence {heap43210}, should be 0")

    def test_heap_get_min_empty_heap(self):
        heap = MinHeap()
        self.assertIsNone(heap.get_min(), f"ERROR: get_min of empty heap returned {heap.get_min()}, should be None")

    def test_heap_remove_min_last_element(self):
        heap = MinHeap()
        for i in heap0:
            heap.insert(i)
        self.assertEqual(heap0, heap.heap, f"get_min cannot be tested because insert created an invalid heap: "
                                           f"{heap.heap} but should be: {heap0}")
        self.assertEqual(0, heap.remove_min(), f"ERROR: heap.remove_min() returned incorrect value ({heap.get_min()}) "
                                               f"for insert sequence {heap0}, should be 0")

    def test_heap_remove_min_without_downheap(self):
        heap = MinHeap()
        for i in heap012:
            heap.insert(i)
        self.assertEqual(heap012, heap.heap, f"get_min cannot be tested because insert created an invalid heap: "
                                             f"{heap.heap} but should be: {heap012}")
        min_rem = heap.remove_min()
        self.assertEqual(0, min_rem, f"ERROR: heap.remove_min() returned incorrect value ({heap.get_min()}) "
                                     f"for insert sequence {heap012}, should be 0")
        self.assertEqual(heap12, heap.heap, f"remove_min failed, was: {heap.heap} but should be: {heap12}")

    def test_heap_remove_min_with_downheap(self):
        heap = MinHeap()
        for i in heap43210:
            heap.insert(i)
        self.assertEqual(heap43210result, heap.heap, f"get_min cannot be tested because insert created an invalid "
                                                     f"heap: {heap.heap} but should be: {heap43210result}")
        min_rem = heap.remove_min()
        self.assertEqual(0, min_rem, f"ERROR: heap.remove_min() returned incorrect value ({heap.get_min()}) "
                                     f"for insert sequence {heap43210result}, should be 0")
        self.assertEqual(heap43210removeMinResult, heap.heap, f"remove_min failed, was: {heap.heap} but should be: "
                                                              f"{heap43210removeMinResult}")

    def test_heap_size_with_insert_and_remove(self):
        heap = MinHeap()
        self.assertEqual(0, heap.get_size(), f"ERROR: get_size returned {heap.get_size()} for empty heap but should "
                                             f"be 0")
        heap.insert(2)
        self.assertEqual(1, heap.get_size(), f"ERROR: get_size returned {heap.get_size()} for heap after one insert "
                                             f"but should be 1")
        heap.insert(2)
        self.assertEqual(2, heap.get_size(), f"ERROR: get_size returned {heap.get_size()} for heap after two inserts "
                                             f"but should be 2")
        heap.remove_min()
        self.assertEqual(1, heap.get_size(), f"ERROR: get_size returned {heap.get_size()} for heap after remove of "
                                             f"one element but should be 1")
        heap.remove_min()
        self.assertEqual(0, heap.get_size(), f"ERROR: get_size returned {heap.get_size()} for empty heap after last "
                                             f"remove_min but should be 0")


if __name__ == "__main__":
    unittest.main()
