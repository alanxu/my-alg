
# Sort

Internal Sort/External Sort

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

### [1481. Least Number of Unique Integers after K Removals](https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/)
```python
class Solution(object):
    def findLeastNumOfUniqueInts(self, arr, k):
        # Create a sorted array of counts of each number.
        # Trick: Add 0 for easier handling corner cases
        counts = [0] + sorted([count for count in Counter(arr).values()])
        C = len(counts)
        
        # Calculate running prefix sum and compare with k
        for i in range(1, C):
            counts[i] += counts[i - 1]
            if counts[i] > k: return C - i
        
        # If reach here, it means all pre sum <= k, so all can be removed
        return 0
```

### [324. Wiggle Sort II](https://leetcode.com/problems/wiggle-sort-ii/)

```python
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        nums.sort()
        mid = (len(nums) - 1) // 2
        # Trick: Wiggle Sort
        #    Both half is reversed
        nums[::2], nums[1::2] = nums[mid::-1], nums[:mid:-1]
```

### [179. Largest Number](https://leetcode.com/problems/largest-number/)

```python
class SortKey(str):
    def __lt__(x, y):
        return x + y > y + x
        
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        ans = ''.join(sorted(map(str, nums), key=SortKey))
        return '0' if ans[0] == '0' else ans
```

### [791. Custom Sort String](https://leetcode.com/problems/custom-sort-string/)

```python
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        order_mp = {}
        for i, x in enumerate(order):
            order_mp[x] = i
        
        s = list(s)
        s.sort(key=lambda x: order_mp[x] if x in order_mp else 26 + ord(x))
        return ''.join(s)
```