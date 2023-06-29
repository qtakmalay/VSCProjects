import unittest

from rabin_karp import RabinKarp

maxPoints = 16.0  # defines the maximum achievable points for the example tested here
points = maxPoints  # stores the actually achieved points based on failed unit tests
summary = ""


def deduct_pts(value):
    global points
    points = points - value
    if points < 0:
        points = 0


def resolve_amount_of_pts_to_deduct(argument):
    pool = {
        "test_rabin_karp_hash_of_text_sequences": 1.5,
        "test_rabin_karp_hash_of_pattern": 2.5,
        "test_is_empty": 0.5,
        "test_kmp_short": 1.0,
        "test_kmp_long": 1.0,
        "test_kmp_special": 1.5,
        "test_kmp_special1": 1.5,
        "test_kmp_special2": 1.5,
        "test_kmp_long_pattern": 2.5,
        "test_kmp_very_long_pattern": 2.5,
    }
    # resolve the pts to deduct from pool
    return pool.get(argument, 0)


class TestAssignment07RabinKarp(unittest.TestCase):
    def setUp(self):
        pass

    ####################################################
    # Definition of test cases
    ####################################################

    def test_rabin_karp_hash_of_pattern(self):
        r = RabinKarp()
        str_pattern = "ef"
        hash_pattern = r.get_rolling_hash_value(str_pattern, '\0', 0)
        self.assertEqual(3031, hash_pattern)

    def test_rabin_karp_hash_of_text_sequences(self):
        r = RabinKarp()
        str_pattern = "abcdef"
        hash_pattern = 0

        hash_pattern = r.get_rolling_hash_value(str_pattern[0:2], '\0', hash_pattern)
        self.assertEqual(2911, hash_pattern, "Wrong hash value. Initial text: abcdef")

        hash_pattern = r.get_rolling_hash_value(str_pattern[1:3], 'a', hash_pattern)
        self.assertEqual(2941, hash_pattern, "Wrong hash value. Initial text: abcdef")

        hash_pattern = r.get_rolling_hash_value(str_pattern[2:4], 'b', hash_pattern)
        self.assertEqual(2971, hash_pattern, "Wrong hash value. Initial text: abcdef")

        hash_pattern = r.get_rolling_hash_value(str_pattern[3:5], 'c', hash_pattern)
        self.assertEqual(3001, hash_pattern, "Wrong hash value. Initial text: abcdef")

        hash_pattern = r.get_rolling_hash_value(str_pattern[4:6], 'd', hash_pattern)
        self.assertEqual(3031, hash_pattern, "Wrong hash value. Initial text: abcdef")

    def test_is_empty(self):
        r = RabinKarp()
        with self.assertRaises(ValueError, msg="ValueError should be raised when looking for pattern in empty strings"):
            r.search("aaa", None)

    def test_kmp_short(self):
        r = RabinKarp()
        res = r.search("xxx", "abcdexxxunbxxxxke")
        self.assertEqual(3, len(res), f"expected the pattern to be found 3 times but was found {len(res)} times")
        self.assertEqual([5, 11, 12], res, f"expected the pattern to be found on positions [5, 11, 12] but was found "
                                           f"on positions {res}")

    def test_kmp_long(self):
        r = RabinKarp()
        res = r.search("is", "Compares two strings lexicographically. The comparison is based on the Unicode value of "
                             "each character in the strings. The character sequence represented by this String object "
                             "is compared lexicographically to the character sequence represented by the argument "
                             "string. The result is a negative integer if this String object lexicographically "
                             "precedes the argument string. The result is a positive integer if this String object "
                             "lexicographically follows the argument string. The result is zero if the strings are "
                             "equal; compareTo returns 0 exactly when the equals(Object) method would return true.")
        self.assertEqual(9, len(res), f"expected the pattern to be found 9 times but was found {len(res)} times")
        self.assertEqual([50, 55, 159, 176, 279, 306, 382, 409, 484], res,
                         f"expected the pattern to be found on positions [50, 55, 159, 176, 279, 306, 382, 409, 484] "
                         f"but was found on positions {res}")

    def test_kmp_special(self):
        r = RabinKarp()
        res = r.search("xxx", "xxxxxA")
        self.assertEqual(3, len(res), f"expected the pattern to be found 3 times but was found {len(res)} times")
        self.assertEqual([0, 1, 2], res, f"expected the pattern to be found on positions [0, 1, 2] but was found on "
                                         f"positions {res}")

    def test_kmp_special1(self):
        r = RabinKarp()
        res = r.search("xxx", "xxxXfl, xxxxA")
        self.assertEqual(3, len(res), f"expected the pattern to be found 3 times but was found {len(res)} times")
        self.assertEqual([0, 8, 9], res, f"expected the pattern to be found on positions [0, 8, 9] but was found on "
                                         f"positions {res}")

    def test_kmp_special2(self):
        r = RabinKarp()
        res = r.search("xyxy", "xXyxyxyfl, xyxxyxyA")
        self.assertEqual(2, len(res), f"expected the pattern to be found 2 times but was found {len(res)} times")
        self.assertEqual([3, 14], res, f"expected the pattern to be found on positions [3, 14] but was found on "
                                       f"positions {res}")

    def test_kmp_long_pattern(self):
        r = RabinKarp()
        res = r.search("ijK lmnopq", "abcdefghijK lmnopqrstuvwxyz, abcdefghijK lmnopqrstuvwxyz.ijk lmnopqr")
        self.assertEqual(2, len(res), f"expected the pattern to be found 2 times but was found {len(res)} times")
        self.assertEqual([8, 37], res, f"expected the pattern to be found on positions [8, 37] but was found on "
                                       f"positions {res}")

    def test_kmp_very_long_pattern(self):
        r = RabinKarp()
        res = r.search(
            "Lorem ips",
            "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore "
            "et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea "
            "rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum "
            "dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore "
            "magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet "
            "clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, "
            "consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam "
            "erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd "
            "gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Duis autem vel eum iriure dolor in "
            "hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero "
            "eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te "
            "feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh "
            "euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis "
            "nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem "
            "vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat "
            "nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril "
            "delenit augue duis dolore te feugait nulla facilisi. Nam liber tempor cum soluta nobis eleifend option "
            "congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, "
            "consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam "
            "erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis "
            "nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit "
            "esse molestie consequat, vel illum dolore eu feugiat nulla facilisis. At vero eos et accusam et justo duo "
            "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. "
            "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore "
            "et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea "
            "rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum "
            "dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos "
            "erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. "
            "sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, "
            "consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam "
            "erat. Consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna "
            "aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita "
            "kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, "
            "consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam "
            "erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd "
            "gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur "
            "sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed "
            "diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea "
            "takimata sanctus. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor "
            "invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo "
            "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. "
            "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore "
            "et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea "
            "rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum "
            "dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore "
            "magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet "
            "clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Duis autem vel eum iriure "
            "dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla "
            "facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit "
            "augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, "
            "sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad "
            "minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo "
            "consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel "
            "illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit "
            "praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Nam liber tempor cum soluta "
            "nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem "
            "ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet "
            "dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper "
            "suscipit lobortis nisl ut aliquip ex ea commodo Lorem ipsum dolor sit amet, consetetur sadipscing elitr, "
            "sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At "
            "vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus "
            "est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy "
            "eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam "
            "et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum "
            "dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor "
            "invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo "
            "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. "
            "Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore "
            "eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum "
            "zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer "
            "adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. "
            "Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex "
            "ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie "
            "consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim "
            "qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Nam liber tempor "
            "cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. "
            "Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut "
            "laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation "
            "ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in "
            "hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis. At "
            "vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus "
            "est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy "
            "eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam "
            "et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum "
            "dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam "
            "dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et "
            "gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit "
            "amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut "
            "labore et dolore magna aliquyam erat. Consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt "
            "ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores "
            "et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem "
            "ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et "
            "dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. "
            "Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit "
            "amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna "
            "aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita "
            "kasd gubergren, no sea takimata sanctus. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed "
            "diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero "
            "eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est "
            "Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy "
            "eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam "
            "et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum "
            "dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor "
            "invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo "
            "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. "
            "Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore "
            "eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum "
            "zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer "
            "adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. "
            "Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex "
            "ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie "
            "consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim "
            "qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Nam liber tempor "
            "cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. "
            "Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut "
            "laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation "
            "ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo.")
        self.assertEqual(52, len(res), f"expected the pattern to be found 52 times but was found {len(res)} times")
        self.assertEqual(
            [0, 268, 296, 564, 592, 860, 1159, 1826, 2345, 2373, 2641, 2669, 2937, 2965, 3342, 3370, 3638, 3666, 3931,
             4199, 4227, 4495, 4523, 4791, 5090, 5757, 6019, 6287, 6315, 6583, 6611, 6879, 7178, 7845, 8364, 8392,
             8660, 8688, 8956, 8984, 9361, 9389, 9657, 9685, 9950, 10218, 10246, 10514, 10542, 10810, 11109, 11776],
            res, f"expected the pattern to be found on positions [0, 268, 296, 564, 592, 860, 1159, 1826, 2345, 2373, "
                 f"2641, 2669, 2937, 2965, 3342, 3370, 3638, 3666, 3931, 4199, 4227, 4495, 4523, 4791, 5090, 5757, "
                 f"6019, 6287, 6315, 6583, 6611, 6879, 7178, 7845, 8364, 8392, 8660, 8688, 8956, 8984, 9361, 9389, "
                 f"9657, 9685, 9950, 10218, 10246, 10514, 10542, 10810, 11109, 11776] but was found on positions {res}")


if __name__ == '__main__':
    unittest.main()
