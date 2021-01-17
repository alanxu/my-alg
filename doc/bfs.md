# Breadth First Search

When you are given an array or a string and some rule that can group the
items, you have a tree/graph structure.

Sometimes you can use memorization to improve performance.

## Problems

### [139. Word Break](https://leetcode.com/problems/word-break/)

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        l = len(s)
        q = collections.deque()
        q.appendleft(0)
        visited = set()
        
        while len(q) > 0:
            i = q.pop()
            
            for j in range(i, l):
                if j in visited:
                    continue
                
                w = s[i:j+1]
                if w in wordDict:
                    q.appendleft(j+1)
                    visited.add(j)
                    
                    if j == l - 1:
                        return True
        return False
```