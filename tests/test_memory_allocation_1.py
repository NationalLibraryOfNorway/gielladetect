import unittest

def b_to_mib(b: int):
    return round(b / 1024 / 1024, 1)

# this needs to be in its own module to avoid conflicting the import system between the tests
class MemoryAllocation1TestCase(unittest.TestCase):
    def test_default_full_classification(self):
        import tracemalloc
        tracemalloc.stop()
        self.assertFalse(tracemalloc.is_tracing(), msg="tracemalloc should not have been started yet")
        tracemalloc.start()
        s1 = tracemalloc.take_snapshot()
        import gielladetect 
        s2 = tracemalloc.take_snapshot()
        self.assertLess(b_to_mib(sum(s.size_diff for s in s2.compare_to(s1, key_type="lineno"))), 1.0)
        d1 = gielladetect.detect("Hva gjør du i dag?") 
        self.assertEqual(d1, "nob")
        s3 = tracemalloc.take_snapshot()
        self.assertGreater(b_to_mib(sum(s.size_diff for s in s3.compare_to(s2, key_type="lineno"))), 180.0)
        d2 = gielladetect.detect("Kva gjer du i dag?") 
        self.assertEqual(d2, "nno")
        s4 = tracemalloc.take_snapshot()
        self.assertLess(b_to_mib(sum(s.size_diff for s in s4.compare_to(s3, key_type="lineno"))), 0.5)
