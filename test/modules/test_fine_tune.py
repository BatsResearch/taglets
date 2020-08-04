from taglets.modules.fine_tune import FineTuneModule
from .test_module import TestModule
import unittest


class TestFineTune(TestModule, unittest.TestCase):
    def _get_module(self, task):
        return FineTuneModule(task)


if __name__ == '__main__':
    unittest.main()
