import unittest

from taglets.modules.zsl_kg_lite import ZSLKGModule
from .test_module import TestModule


class TestZSLKGModule(TestModule, unittest.TestCase):
    def _get_module(self, task):
        return ZSLKGModule(task)


if __name__ == '__main__':
    unittest.main()