from taglets.modules.prototype import PrototypeModule
from .test_module import TestModule
import unittest


class TestPrototype(TestModule, unittest.TestCase):

    def _get_module(self, task):
        mod = PrototypeModule(task)
        # lower meta-params to reduce train time
        for taglet in mod.taglets:
            taglet.set_train_shot(shot=1)
            taglet.set_train_way(way=3)
            taglet.set_query(query=2)
        return mod


if __name__ == '__main__':
    unittest.main()
