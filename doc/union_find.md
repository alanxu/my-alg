

### [684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)

```python
class DSU:
    def __init__(self):
        self.par = [i for i in range(1001)]
        self.rnk = [0] * 1001
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        elif self.rnk[xr] < self.rnk[yr]:
            self.par[xr] = yr
        elif self.rnk[xr] > self.rnk[yr]:
            self.par[yr] = xr
        else:
            self.par[xr] = yr
            self.rnk[yr] += 1
        return True
    
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        dsu = DSU()
        for edge in edges:
            if not dsu.union(*edge):
                return edge
```

