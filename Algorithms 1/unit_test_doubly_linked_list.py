import unittest

from my_sorted_doubly_linked_list import MySortedDoublyLinkedList
from my_list_node import MyListNode

maxPoints = 16.0  # defines the maximum achievable points for the example tested here
points = maxPoints  # stores the actually achieved points based on failed unit tests
summary = ""


def create_list_from_array(arr):
    head = MyListNode(arr[0]) if arr else None
    old_node = head
    for i in range(1, len(arr)):
        new_node = MyListNode(arr[i])
        new_node.prev_node = old_node
        old_node.next_node = new_node
        old_node = new_node
    return MySortedDoublyLinkedList(head, old_node, len(arr))


def deduct_pts(value):
    global points
    points = points - value
    if points < 0:
        points = 0


def resolve_amount_of_pts_to_deduct(argument):
    pool = {
        "test_get_value": 1.0,
        "test_invalid_get_value": 0.25,
        "test_search_value": 1.0,
        "test_invalid_search_value": 0.25,
        "test_insert": 2.0,
        "test_invalid_insert": 0.25,
        "test_remove_first": 2.0,
        "test_no_remove_first": 0.25,
        "test_invalid_remove_first": 0.25,
        "test_remove_all": 2.0,
        "test_no_remove_all": 0.25,
        "test_invalid_remove_all": 0.25,
        "test_remove_duplicates": 2.0,
        "test_filter_n_max": 1.5,
        "test_invalid_filter_n_max": 0.25,
        "test_filter_odd": 2.5,
    }
    # resolve the pts to deduct from pool
    return pool.get(argument, 0)


class UnitTestSortedDoublyLinkedList(unittest.TestCase):
    def setUp(self):
        self.arr_base = [1, 2, 3, 4]
        self.arr_fib = [1, 1, 2, 3, 5, 8, 13, 21]
        self.arr_even = [-2, -2, -2, 0, 2, 2, 4, 6, 8, 8]
        self.arr_odd = [-1, 1, 1, 1, 3, 3, 5, 7, 9]
        self.empty_dll = create_list_from_array([])
        self.sorted_dll = create_list_from_array(self.arr_base)

    ####################################################
    # Definition of test cases
    ####################################################

    def test_get_value(self):
        self.assertEqual(1, self.sorted_dll.get_value(0), "Wrong return value for index 0.")
        self.assertEqual(2, self.sorted_dll.get_value(1), "Wrong return value for index 1.")
        print(self.sorted_dll.get_value(1))
        self.assertEqual(3, self.sorted_dll.get_value(2), "Wrong return value for index 2.")
        self.assertEqual(4, self.sorted_dll.get_value(3), "Wrong return value for index 3.")

    def test_invalid_get_value(self):
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.empty_dll.get_value(0)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.get_value(-1)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.get_value(4)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.get_value(2.5)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.get_value('a')

    def test_search_value(self):
        self.assertEqual(0, self.sorted_dll.search_value(1), "Wrong return value for value 1.")
        self.assertEqual(1, self.sorted_dll.search_value(2), "Wrong return value for value 2.")
        self.assertEqual(2, self.sorted_dll.search_value(3), "Wrong return value for value 3.")
        self.assertEqual(3, self.sorted_dll.search_value(4), "Wrong return value for value 4.")

    def test_no_search_value(self):
        self.assertEqual(-1, self.empty_dll.search_value(0), "Wrong return value for value 0.")
        self.assertEqual(-1, self.sorted_dll.search_value(0), "Wrong return value for value 0.")
        self.assertEqual(-1, self.sorted_dll.search_value(5), "Wrong return value for value 5.")

    def test_invalid_search_value(self):
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.get_value(2.5)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.get_value('a')

    def test_insert(self):
        self.empty_dll.insert(0)
        self.assertEqual("[0]", str(self.empty_dll), "Insert operation failed.")
        self.assertEqual(1, len(self.empty_dll), "Size incorrect after insert operation.")
        self.sorted_dll.insert(3)
        self.assertEqual("[1, 2, 3, 3, 4]", str(self.sorted_dll), "Insert operation failed.")
        self.assertEqual(5, len(self.sorted_dll), "Size incorrect after insert operation.")
        self.sorted_dll.insert(5)
        self.assertEqual("[1, 2, 3, 3, 4, 5]", str(self.sorted_dll), "Insert operation at tail failed.")
        self.assertEqual(6, len(self.sorted_dll), "Size incorrect after insert operation at tail.")
        self.sorted_dll.insert(0)
        self.assertEqual("[0, 1, 2, 3, 3, 4, 5]", str(self.sorted_dll), "Insert operation at head failed.")
        self.assertEqual(7, len(self.sorted_dll), "Size incorrect after insert operation at head.")

    def test_invalid_insert(self):
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.insert(2.5)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.insert('a')

    def test_remove_first(self):
        self.assertTrue(self.sorted_dll.remove_first(3), "Wrong return value for remove first operation.")
        self.assertEqual("[1, 2, 4]", str(self.sorted_dll), "Remove first operation failed.")
        self.assertEqual(3, len(self.sorted_dll), "Size incorrect after remove first operation.")
        self.assertTrue(self.sorted_dll.remove_first(4), "Wrong return value for remove first operation.")
        self.assertEqual("[1, 2]", str(self.sorted_dll), "Remove first operation at tail failed.")
        self.assertEqual(2, len(self.sorted_dll), "Size incorrect after remove first operation at tail.")
        self.assertTrue(self.sorted_dll.remove_first(1), "Wrong return value for remove first operation.")
        self.assertEqual("[2]", str(self.sorted_dll), "Remove first operation at head failed.")
        self.assertEqual(1, len(self.sorted_dll), "Size incorrect after remove first operation at head.")
        self.assertTrue(self.sorted_dll.remove_first(2), "Wrong return value for remove first operation.")
        self.assertEqual("[]", str(self.sorted_dll), "Remove first operation failed.")

    def test_no_remove_first(self):
        self.assertFalse(self.empty_dll.remove_first(0), "Wrong return value for remove first operation.")
        self.assertFalse(self.sorted_dll.remove_first(5), "Wrong return value for remove first operation.")
        self.assertEqual("[1, 2, 3, 4]", str(self.sorted_dll),
                         "Remove first operation with not contained value failed.")
        self.assertEqual(4, len(self.sorted_dll),
                         "Size incorrect after remove first operation with not contained value.")

    def test_invalid_remove_first(self):
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.remove_first(2.5)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.remove_first('a')

    def test_remove_all(self):
        self.sorted_dll = create_list_from_array(self.arr_even)
        self.assertTrue(self.sorted_dll.remove_all(2), "Wrong return value for remove all operation.")
        self.assertEqual("[-2, -2, -2, 0, 4, 6, 8, 8]", str(self.sorted_dll), "Remove all operation failed.")
        self.assertEqual(8, len(self.sorted_dll), "Size incorrect after remove all operation.")
        self.assertTrue(self.sorted_dll.remove_all(8), "Wrong return value for remove all operation.")
        self.assertEqual("[-2, -2, -2, 0, 4, 6]", str(self.sorted_dll), "Remove all operation at tail failed.")
        self.assertEqual(6, len(self.sorted_dll), "Size incorrect after remove all operation at tail.")
        self.assertTrue(self.sorted_dll.remove_all(-2), "Wrong return value for remove all operation.")
        self.assertEqual("[0, 4, 6]", str(self.sorted_dll), "Remove all operation at head failed.")
        self.assertEqual(3, len(self.sorted_dll), "Size incorrect after remove all operation at head.")
        self.assertTrue(self.sorted_dll.remove_all(4), "Wrong return value for remove all operation.")
        self.assertEqual("[0, 6]", str(self.sorted_dll), "Remove all operation failed.")
        self.assertEqual(2, len(self.sorted_dll), "Size incorrect after remove all operation.")

    def test_no_remove_all(self):
        self.assertFalse(self.empty_dll.remove_all(0), "Wrong return value for remove all operation.")
        self.assertFalse(self.sorted_dll.remove_all(5), "Wrong return value for remove all operation.")
        self.assertEqual("[1, 2, 3, 4]", str(self.sorted_dll), "Remove all operation with not contained value failed.")
        self.assertEqual(4, len(self.sorted_dll), "Size incorrect after remove all operation with not contained value.")

    def test_invalid_remove_all(self):
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.remove_all(2.5)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.remove_all('a')

    def test_remove_duplicates(self):
        self.empty_dll.remove_duplicates()
        self.assertEqual("[]", str(self.empty_dll), "Remove duplicates operation failed.")
        self.assertEqual(0, len(self.empty_dll), "Size incorrect after remove duplicates operation.")
        self.sorted_dll.remove_duplicates()
        self.assertEqual("[1, 2, 3, 4]", str(self.sorted_dll), "Remove duplicates operation failed.")
        self.assertEqual(4, len(self.sorted_dll), "Size incorrect after remove duplicates operation.")
        self.sorted_dll = create_list_from_array(self.arr_fib)
        self.sorted_dll.remove_duplicates()
        self.assertEqual("[1, 2, 3, 5, 8, 13, 21]", str(self.sorted_dll), "Remove duplicates operation failed.")
        self.assertEqual(7, len(self.sorted_dll), "Size incorrect after remove duplicates operation.")
        self.sorted_dll = create_list_from_array(self.arr_even)
        self.sorted_dll.remove_duplicates()
        self.assertEqual("[-2, 0, 2, 4, 6, 8]", str(self.sorted_dll), "Remove duplicates operation failed.")
        self.assertEqual(6, len(self.sorted_dll), "Size incorrect after remove duplicates operation.")
        self.sorted_dll = create_list_from_array(self.arr_odd)
        self.sorted_dll.remove_duplicates()
        self.assertEqual("[-1, 1, 3, 5, 7, 9]", str(self.sorted_dll), "Remove duplicates operation failed.")
        self.assertEqual(6, len(self.sorted_dll), "Size incorrect after remove duplicates operation.")

    def test_filter_n_max(self):
        self.sorted_dll.filter_n_max(4)
        self.assertEqual("[1, 2, 3, 4]", str(self.sorted_dll), "Filter n max operation failed.")
        self.assertEqual(4, len(self.sorted_dll), "Size incorrect after filter n max operation.")
        self.sorted_dll.filter_n_max(3)
        self.assertEqual("[2, 3, 4]", str(self.sorted_dll), "Filter n max operation failed.")
        self.assertEqual(3, len(self.sorted_dll), "Size incorrect after filter n max operation.")
        self.sorted_dll.filter_n_max(2)
        self.assertEqual("[3, 4]", str(self.sorted_dll), "Filter n max operation failed.")
        self.assertEqual(2, len(self.sorted_dll), "Size incorrect after filter n max operation.")
        self.sorted_dll.filter_n_max(1)
        self.assertEqual("[4]", str(self.sorted_dll), "Filter n max operation failed.")
        self.assertEqual(1, len(self.sorted_dll), "Size incorrect after filter n max operation.")
        self.sorted_dll = create_list_from_array(self.arr_fib)
        self.sorted_dll.filter_n_max(8)
        self.assertEqual("[1, 1, 2, 3, 5, 8, 13, 21]", str(self.sorted_dll), "Filter n max operation failed.")
        self.assertEqual(8, len(self.sorted_dll), "Size incorrect after filter n max operation.")
        self.sorted_dll = create_list_from_array(self.arr_even)
        self.sorted_dll.filter_n_max(2)
        self.assertEqual("[8, 8]", str(self.sorted_dll), "Filter n max operation failed.")
        self.assertEqual(2, len(self.sorted_dll), "Size incorrect after filter n max operation.")
        self.sorted_dll = create_list_from_array(self.arr_odd)
        self.sorted_dll.filter_n_max(1)
        self.assertEqual("[9]", str(self.sorted_dll), "Filter n max operation failed.")
        self.assertEqual(1, len(self.sorted_dll), "Size incorrect after filter n max operation.")

    def test_invalid_filter_n_max(self):
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.empty_dll.filter_n_max(1)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.filter_n_max(0)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.filter_n_max(5)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.filter_n_max(2.5)
        with self.assertRaises(ValueError, msg="No ValueError for invalid input."):
            self.sorted_dll.filter_n_max('a')

    def test_filter_odd(self):
        self.empty_dll.filter_odd()
        self.assertEqual("[]", str(self.empty_dll), "Filter odd operation failed.")
        self.assertEqual(0, len(self.empty_dll), "Size incorrect after filter odd operation.")
        self.sorted_dll.filter_odd()
        self.assertEqual("[1, 3]", str(self.sorted_dll), "Filter odd operation failed.")
        self.assertEqual(2, len(self.sorted_dll), "Size incorrect after filter odd operation.")
        self.sorted_dll = create_list_from_array(self.arr_fib)
        self.sorted_dll.filter_odd()
        self.assertEqual("[1, 1, 3, 5, 13, 21]", str(self.sorted_dll), "Filter odd operation failed.")
        self.assertEqual(6, len(self.sorted_dll), "Size incorrect after filter odd operation.")
        self.sorted_dll = create_list_from_array(self.arr_even)
        self.sorted_dll.filter_odd()
        self.assertEqual("[]", str(self.sorted_dll), "Filter odd operation failed.")
        self.assertEqual(0, len(self.sorted_dll), "Size incorrect after filter odd operation.")
        self.sorted_dll = create_list_from_array(self.arr_odd)
        self.sorted_dll.filter_odd()
        self.assertEqual("[-1, 1, 1, 1, 3, 3, 5, 7, 9]", str(self.sorted_dll), "Filter odd operation failed.")
        self.assertEqual(9, len(self.sorted_dll), "Size incorrect after filter odd operation.")


if __name__ == "__main__":
    unittest.main()

