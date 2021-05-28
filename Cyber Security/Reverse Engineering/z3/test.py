#!/usr/bin/python

from z3 import *

key1, key2, key3 = BitVecs('key1 key2 key3', 128)
s = Solver()
s.add(key1 == key2)
s.add(key2 == key3)
s.add(key1==2)
s.check()
m = s.model()
print(m, m[key1], m[key2], m[key3])
