

### [493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/)

```python
class Solution:
    
    class Bitree:
        def __init__(self, n):
            self.n = n + 1
            self.sums = [0] * self.n
            
        def update(self, i, v):
            while i < self.n:
                self.sums[i] += v
                i += (i & -i)
        
        def query(self, i):
            s = 0
            while i > 0:
                s += self.sums[i]
                i -= (i & -i)
            return s
    
    def reversePairs(self, nums: List[int]) -> int:
        new_nums = nums + [2 * n for n in nums]
        sorted_nums = sorted(list(set(new_nums)))
        rank = {}
        for i, n in enumerate(sorted_nums):
            # The rank has to be > 0
            rank[n] = i + 1
            
        bitree = self.Bitree(len(sorted_nums))
        
        ans = 0
        for n in reversed(nums):
            ans += bitree.query(rank[n] - 1)
            bitree.update(rank[n * 2], 1)
            
        return ans
```

### [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

