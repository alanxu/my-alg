

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

### [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/)

```python
class SegmentTreeNode:
	def __init__(self,low,high):
		self.low = low
		self.high = high
		self.left = None
		self.right = None
		self.cnt = 0        
        
class Solution: 
    def _bulid(self, left, right): 
        root = SegmentTreeNode(self.cumsum[left],self.cumsum[right])
        if left == right:
            return root
        
        mid = (left+right)//2
        root.left = self._bulid(left, mid)
        root.right = self._bulid(mid+1, right)
        return root
    
    def _update(self, root, val):
        if not root:
            return 
        if root.low<=val<=root.high:
            root.cnt += 1
            self._update(root.left, val)
            self._update(root.right, val)
                  
    def _query(self, root, lower, upper):
        if lower <= root.low and root.high <= upper:
            return root.cnt
        if upper < root.low or root.high < lower:
            return 0
        return self._query(root.left, lower, upper) + self._query(root.right, lower, upper)
        
    # prefix-sum + SegmentTree | O(nlogn)      
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        cumsum = [0]
        for n in nums:
            cumsum.append(cumsum[-1]+n)
            
        self.cumsum = sorted(list(set(cumsum))) # need sorted
        root = self._bulid(0,len(self.cumsum)-1)
        
        res = 0
        for csum in cumsum:
            res += self._query(root, csum-upper, csum-lower)
            self._update(root, csum)  

        print(self._query(root, 0, 4))

        return res
```