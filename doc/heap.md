

## Problems

### [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)

```python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # Trick: Use Counter to create {value: count}
        count = Counter(words)
        heap = [(-freq, word) for word, freq in count.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]
```