#
https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/discuss/49708/Sliding-Window-algorithm-template-to-solve-all-the-Leetcode-substring-search-problem.


### [395. Longest Substring with At Least K Repeating Characters](https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/)

```python
class Solution:
    def longestSubstring(self, s, k):
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(t, k) for t in s.split(c))
    
        return len(s)
```

### [95. Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/)

```python
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        @functools.lru_cache(None)
        def dfs(start, end):
            # Thought about use start == end as termination
            # condition, to use that, you still needs to 
            # handle end > start, say for s,...,e, when root
            # is s, we need to get left, we call dfs(s, s - 1),
            # this call needs to return [None], we can handle
            # it at end but it is messy.
            if start > end:
                return [None]
            
            ans = []
            for root in range(start, end + 1):
                lefts = dfs(start, root - 1)
                rights = dfs(root + 1, end)
                
                for l in lefts:
                    for r in rights:
                        tree = TreeNode(root, l, r)
                        ans.append(tree)
            return ans
        return dfs(1, n)
```