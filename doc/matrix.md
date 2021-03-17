# Matrix

Use row - col to locate diagonals for each cell



## Problems

### [1329. Sort the Matrix Diagonally](https://leetcode.com/problems/sort-the-matrix-diagonally/)

```python
from heapq import *
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        
        diagonals = defaultdict(list)
        
        R, C = len(mat), len(mat[0])
        
        # Trick: Use row - col to locate diagonals for each cell
        for row in range(R):
            for col in range(C):
                heappush(diagonals[row - col], mat[row][col])
                
        for row in range(R):
            for col in range(C):
                mat[row][col] = heappop(diagonals[row - col])
                
        return mat
```