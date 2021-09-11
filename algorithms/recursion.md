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





### [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

### [139. Word Break](https://leetcode.com/problems/word-break/)

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Time: O(n^3) - The recursive call is n^2, slicing is n
        # to splic a string to any parts (1..n)
        # Space: O(n) - s as para for recursive call, always less than n
        wordDict = set(wordDict)
        @functools.lru_cache(None)
        def _wordBreak(s: str) -> bool:
            if not s:
                return True
            for i in range(1, len(s) + 1):
                w = s[:i]
                if w in wordDict and _wordBreak(s[i:]):
                    return True
            return False
        return _wordBreak(s)
```

### [454. 4Sum II](https://leetcode.com/problems/4sum-ii/)

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        """
        Alg: Recursion
        Use recursion to generate combination of nums in different lists. Do it at the last
        recursion (tree leaves).
        
        Intuition: Divide lists into 2 parts. Calc sum of all combinations of nums in each list
        in each part. For first part, same all sum as key and its counts. For sums in second part,
        check how many combinations with -sum. Add up to final ans.
        
        Time to calculate sum of combinations is n^k (n is len of a list, k is num of lists),
        so if we can divide the 2 group evenly, we can reduct time complexity the best.
        
        Time:  O(n^(k//2))
        Space: O(n^(k//2))
        """
        L, nums = 4, [nums1, nums2, nums3, nums4]
        mp = defaultdict(int)
        self.ans = 0
        
        def process_first_grp(i, _sum):
            if i == L // 2:
                mp[_sum] += 1
                return
            for x in nums[i]:
                process_first_grp(i + 1, _sum + x)
        
        def process_second_grp(i, _sum):
            if i == L:
                self.ans += mp[-_sum]
                return
            for x in nums[i]:
                process_second_grp(i + 1, _sum + x)
                
        process_first_grp(0, 0)
        process_second_grp(L // 2, 0)
        return self.ans
```