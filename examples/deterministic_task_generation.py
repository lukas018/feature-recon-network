#/usr/bin/env python3

"""Demonstrates how to do deterministic task generation using l2l"""
import random

def fixed_random(func):
    """Create the data"""
    def _func(self, i):
        state = random.getstate()
        if self.deterministic or self.seed is not None:
            random.seed(self.seed + i)
            results = func(self, i)
            random.setstate(state)
        else:
            results = func(self, i)
        return results

    return _func


class RandomTest:

    def __init__(self, seed=42, deterministic=False):
        self.seed = seed
        self.deterministic = deterministic

    @fixed_random
    def test_function(self, i):
        return [random.randint(0, 10) for x in range(10)]


rt = RandomTest(0)
print(rt.test_function(0))
print(rt.test_function(0))
rt.seed = 1
print(rt.test_function(0))
print(rt.test_function(0))
