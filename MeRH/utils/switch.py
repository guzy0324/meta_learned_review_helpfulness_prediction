from __future__ import annotations
from typing import Mapping

# https://www.zhihu.com/question/50498770/answer/121232273
class Switch:
    def __init__(self, init: Mapping = {}):
        self.branches = {k: init[k] for k in init}

    def __call__(self, case):
        def decorator(func):
            self.branches[case] = func
            return func

        return decorator

    def __getitem__(self, case):
        return self.branches[case]

    def __iter__(self):
        return iter(self.branches)

    def __len__(self):
        return len(self.branches)

    def __getattr__(self, attr):
        return getattr(self.branches, attr)

if __name__ == '__main__':
    switch = Switch(Switch())

    @switch("a")
    def a():
        pass

    print(len(switch))
    print(switch.items())
    print(switch["a"] is a)
