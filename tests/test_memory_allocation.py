import unittest

def b_to_mib(b: int):
    return round(b / 1024 / 1024, 1)

class MemoryAllocationTestCase(unittest.TestCase):
    def test_import(self):
        import tracemalloc
        tracemalloc.stop()
        self.assertFalse(tracemalloc.is_tracing(), msg="tracemalloc should not have been started yet")
        tracemalloc.start()
        s1 = tracemalloc.take_snapshot()
        # self.assertEqual(malloc_before_import, 1224)
        import gielladetect
        s2 = tracemalloc.take_snapshot()
        diff = s2.compare_to(s1, key_type="lineno")
        self.assertGreaterEqual(b_to_mib(sum((s.size_diff for s in diff))), 1.0)
        
