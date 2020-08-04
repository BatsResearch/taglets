from taglets.modules.transfer import TransferModule
from .test_module import TestModule
import unittest


class TestTransfer(TestModule, unittest.TestCase):
    def _get_module(self, task):
        return TransferModule(task)


if __name__ == '__main__':
    unittest.main()
