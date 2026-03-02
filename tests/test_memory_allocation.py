import unittest

def b_to_mib(b: int):
    return round(b / 1024 / 1024, 1)

class MemoryAllocationTestCase(unittest.TestCase):
    def test_memory_allocation_on_import(self):
        import tracemalloc
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        malloc_before_import = sum((s.size for s in snapshot1.statistics(key_type="lineno")))
        self.assertEqual(malloc_before_import, 1224)
        import gielladetect
        snapshot2 = tracemalloc.take_snapshot()
        diff = snapshot2.compare_to(snapshot1, key_type="lineno")
        malloc_after_import_diff = sum((s.size for s in diff))
        self.assertAlmostEqual(b_to_mib(malloc_after_import_diff), 187.6)
        
