2 ^ 10 = 1024

Time complexity:
a in s
s[i:j] take O(j-i)

if N > 10000, we can just use linear or logN algo

x % (n|1), if n is even n + 1, if n is odd n

heapq
Counter
enumerate()
str.isdigit() cannot work with '-11'

tuple can be hashed, but list cannot.


https://docs.python.org/2/library/string.html#format-specification-mini-language


import re
ss = re.split('\+|-|\*', s)
print(ss)

tokens = re.split('(\D)', input)


for/else control flow

i = bisect.bisect(dp, [s + 1]) - 1


# Math
numerator/denominator
LCM - Least Common Multiple
GCF - Greatest Common Factor
Catalan Numbers


# Where something is wrong
* Use [[] * n for _ in range(m)] rather than [[] * n] * m
* Check borders
* Loop wont exit, did you add to visited?
* When you property name is wrong, it will not error out  node.pre = node vs. node.prev = node
* Did u mixed up with = and ==?

# Terminology
Referenced array