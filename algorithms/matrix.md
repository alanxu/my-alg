# Matrix

Use row - col to locate diagonals for each cell



## Spiral Matrix

### [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        direction = 0
        step = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = 0, 0
        visited = [[False] * n for _ in range(m)]
        ans = []
        
        for _ in range(m * n):
            ans.append(matrix[x][y])
            visited[x][y] = True
            
            nx, ny = x + step[direction][0], y + step[direction][1]
            if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
                x, y = nx, ny
            else:
                direction = (direction + 1) % 4
                x, y = x + step[direction][0], y + step[direction][1]
                
        return ans
```

### [59. Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)

```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        # Trick: for non prim data type, don't use * n
        ans = [[0] * n for _ in range(n)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        row, col = 0, 0
        d, i = 0, 1
        visited = set()
        while i <= n * n:
            # print((i, row, col))
            ans[row][col] = i
            visited.add((row, col))
            
            while  i < n * n:
                r = row + directions[d][0]
                c = col + directions[d][1]
            
                if r > n - 1 or r < 0 or c > n - 1 or c < 0 or (r, c) in visited:
                    d = (d + 1) % 4
                else:
                    row, col = r, c
                    break

            i += 1

        return ans
```

## Sorted Matrix

### [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)


## Others

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

### [48. Rotate Image](https://leetcode.com/problems/rotate-image/)

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        def rotate_pos_clockwise(i,j):
            return j,self.n-1-i
        
        n = len(matrix)
        self.n = n
        if(n<=1): return
        for i in range((n+1)//2):
            for j in range(i,n-i-1):
                up_val = matrix[i][j]
                i_right,j_right = rotate_pos_clockwise(i,j)
                i_bottom,j_bottom = rotate_pos_clockwise(i_right,j_right)
                i_left,j_left = rotate_pos_clockwise(i_bottom,j_bottom)
                matrix[i][j] = matrix[i_left][j_left]
                matrix[i_left][j_left] = matrix[i_bottom][j_bottom]
                matrix[i_bottom][j_bottom] = matrix[i_right][j_right]
                matrix[i_right][j_right] = up_val
```

### [1041. Robot Bounded In Circle](https://leetcode.com/problems/robot-bounded-in-circle/)

```python
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        # N: 0, E: 1, S: 2, W: 3
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Initial Position
        x = y = 0
        
        # Initial facing north
        idx = 0
        
        for i in instructions:
            if i == 'L':
                idx = (idx + 3) % 4
            elif i == 'R':
                idx = (idx + 1) % 4
            else:
                x += directions[idx][0]
                y += directions[idx][1]
                
        return (x == 0 and y == 0) or idx != 0
```