import unittest

def b_to_mib(b: int):
    return round(b / 1024 / 1024, 1)

class MemoryAllocation2TestCase(unittest.TestCase):
    def test_choosing_languages_consumes_less_memory(self):
        import tracemalloc
        tracemalloc.stop()
        self.assertFalse(tracemalloc.is_tracing(), msg="tracemalloc should not have been started yet")
        tracemalloc.start()
        s1 = tracemalloc.take_snapshot()
        import gielladetect 
        s2 = tracemalloc.take_snapshot()
        self.assertLess(b_to_mib(sum(s.size_diff for s in s2.compare_to(s1, key_type="lineno"))), 1.0)
        chosen_langs = ["nob", "nno"]
        classifier = gielladetect.text_cat.Classifier(langs=chosen_langs)
        s3 = tracemalloc.take_snapshot()
        self.assertLess(b_to_mib(sum(s.size_diff for s in s3.compare_to(s2, key_type="lineno"))), 20.0)
        d1 = classifier.classify("Hva gjør du i dag?") # TODO: langs
        self.assertEqual(d1, "nob")
        d2 = classifier.classify("Kva gjer du i dag?") # TODO: langs
        self.assertEqual(d2, "nno")
        s4 = tracemalloc.take_snapshot()
        self.assertLess(b_to_mib(sum(s.size_diff for s in s4.compare_to(s3, key_type="lineno"))), 0.5)
        self.assertEqual(classifier.langs, set(chosen_langs))
        self.assertEqual(classifier.langs_warned, set())

    def test_default_classifier_is_cached(self):
        import gielladetect
        gielladetect.text_cat._default_classifier.cache_clear()
        cache1 = gielladetect.text_cat._default_classifier.cache_info()
        self.assertEqual(cache1.currsize, 0)
        self.assertEqual(cache1.hits, 0)
        self.assertEqual(cache1.misses, 0)
        self.assertEqual(cache1.maxsize, 1)
        self.assertEqual(gielladetect.text_cat._default_classifier.cache_info().hits, 0)
        c1 = gielladetect.text_cat._default_classifier()
        cache2 = gielladetect.text_cat._default_classifier.cache_info()
        self.assertEqual(cache2.currsize, 1)
        self.assertEqual(cache2.hits, 0)
        self.assertEqual(cache2.misses, 1)
        self.assertEqual(cache2.maxsize, 1)
        c2 = gielladetect.text_cat._default_classifier()
        cache3 = gielladetect.text_cat._default_classifier.cache_info()
        self.assertEqual(cache3.currsize, 1)
        self.assertEqual(cache3.hits, 1)
        self.assertEqual(cache3.misses, 1)
        self.assertEqual(cache3.maxsize, 1)
        self.assertIs(c1, c2)