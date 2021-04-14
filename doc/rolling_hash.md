# Multi Patter Search
Multiple pattern search in a string. All such problems usually could be solved by sliding window approach in 
a linear time. The challenge here is how to implement constant-time slice to fit into this linear time.

If the patterns are not known in advance, i.e. it's "find duplicates" problem, one could use one of two ways 
to implement constant-time slice: Bitmasks or Rabin-Karp. Please check article Repeated DNA Sequences 
for the detailed comparison of these two algorithms.


### [187. Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/)


### [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        Ls, Lp = len(s), len(p)
        
        if Lp > Ls:
            return []
        
        p_count, s_count = [0] * 26, [0] * 26
        for ch in p:
            p_count[ord(ch) - ord('a')] += 1
        
        ans = []
        
        # Trick: Sliding window
        for right in range(Ls):
            s_count[ord(s[right]) - ord('a')] += 1
            
            # When window is longer than Lp
            if right >= Lp:
                # 
                s_count[ord(s[right - Lp]) - ord('a')] -= 1
                
            if p_count == s_count:
                ans.append(right - Lp + 1)
            
            
        return ans
```

### [1044. Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/)