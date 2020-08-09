from taglets.modules.multitask import MultiTaskModule
from .test_module import TestModule
import unittest


class TestTransfer(TestModule, unittest.TestCase):
    def _get_module(self, task):
        return MultiTaskModule(task)


if __name__ == '__main__':
    unittest.main()
