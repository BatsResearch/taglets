from taglets.modules.prototype import PrototypeModule
from .test_module import TestModule
import unittest


class TestPrototype(TestModule, unittest.TestCase):
    def _get_module(self, task):
        return PrototypeModule(task)


if __name__ == '__main__':
    unittest.main()
