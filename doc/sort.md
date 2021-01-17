

## Problems

### [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)
```python
class Solution(object):
    def topKFrequent(self, words, k):
        count = collections.Counter(words)
        candidates = count.keys()
        candidates.sort(key = lambda w: (-count[w], w))
        return candidates[:k]
```