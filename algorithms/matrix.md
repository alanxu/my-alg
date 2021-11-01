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


### [353. Design Snake Game](https://leetcode.com/problems/design-snake-game/)

```python
class SnakeGame:

    def __init__(self, width: int, height: int, food: List[List[int]]):
        """
        Initialize your data structure here.
        @param width - screen width
        @param height - screen height 
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].
        """
        self.snake = deque([(0, 0)])
        self.cols, self.rows = width, height
        self.food = list(reversed(food))
        self.score = 0
        self.dirs = {
            'L': (0, -1),
            'R': (0, 1),
            'U': (-1, 0),
            'D': (1, 0)
        }
        
    def move(self, direction: str) -> int:
        """
        Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down 
        @return The game's score after the move. Return -1 if game over. 
        Game over when snake crosses the screen boundary or bites its body.
        """
        
        # Calc new location
        r, c = self.snake[0][0] + self.dirs[direction][0], self.snake[0][1] + self.dirs[direction][1]
        
        # Check new location not exceed border
        if 0 > r or r >= self.rows or 0 > c or c >= self.cols:
            return -1
        
        
        if not self.food or [r, c] != self.food[-1]:
            # If it is normal empty cell, move snake by removing tail
            self.snake.pop()
        else:
            # If it is food, head move to new location, tail remains,
            # score increment
            self.food.pop()
            self.score += 1
        
        # Check occupy snake body
        if (r, c) in self.snake:
            return -1
        
        # Add new pos after check occupy
        self.snake.appendleft((r, c))
        
        return self.score


# Your SnakeGame object will be instantiated and called as such:
# obj = SnakeGame(width, height, food)
# param_1 = obj.move(direction)
```


### [1515. Best Position for a Service Centre](https://leetcode.com/problems/best-position-for-a-service-centre/)

```python
# Fav
class Solution:    
    def getMinDistSum(self, positions: List[List[int]]) -> float:
        # Alg: Geometric median - Binary Search
        # https://youtu.be/8IX9j5WLLD4
        fn = lambda x, y: sum(sqrt((x-xx)**2 + (y-yy)**2) for xx, yy in positions)
        #centroid as starting point
        x = sum(x for x, _ in positions)/len(positions)
        y = sum(y for _, y in positions)/len(positions)
        
        ans = fn(x, y)
        span = 100
        
        while span > 1e-6:
            zoom = True
            for dx, dy in (1, 0), (-1, 0), (0, 1), (0, -1):
                # Trick: Search matrix with dynamic granularity
                _x, _y = x + dx * span, y + dy * span
                d = fn(_x, _y)
                if d < ans:
                    ans = d
                    x, y = _x, _y
                    zoom = False
                    break
            if zoom: span /= 2
        return ans
    
```

### [289. Game of Life](https://leetcode.com/problems/game-of-life/)

```python
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows, cols = len(board), len(board[0])
        for r in range(rows):
            for c in range(cols):
                lives = 0
                for _r, _c in (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1), \
                (r + 1, c - 1), (r - 1, c + 1), (r + 1, c + 1), (r - 1, c - 1):
                    if 0 <= _r < rows and 0 <= _c < cols and board[_r][_c] in (1, -1):
                        lives += 1
                if board[r][c] == 1 and (lives < 2 or lives > 3):
                    board[r][c] = -1
                if board[r][c] == 0 and lives == 3:
                    board[r][c] = 2
        
        for r in range(rows):
            for c in range(cols):  
                if board[r][c] > 0:
                    board[r][c] = 1
                else:
                    board[r][c] = 0
```


### [1428. Leftmost Column with at Least a One](https://leetcode.com/problems/leftmost-column-with-at-least-a-one/)

```python
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        
        rows, cols = binaryMatrix.dimensions()
        
        # Set pointers to the top-right corner.
        current_row = 0
        current_col = cols - 1
        
        # Repeat the search until it goes off the grid.
        while current_row < rows and current_col >= 0:
            if binaryMatrix.get(current_row, current_col) == 0:
                current_row += 1
            else:
                current_col -= 1
        
        # If we never left the last column, it must have been all 0's.
        return current_col + 1 if current_col != cols - 1 else -1
```



### [766. Toeplitz Matrix](https://leetcode.com/problems/toeplitz-matrix/)

```python
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        rows, cols = len(matrix), len(matrix[0])
        def check(r, c, val):
            if matrix[r][c] != val:
                return False
            _r, _c = r + 1, c + 1
            if 0 <= _r < rows and 0 <= _c < cols:
                return check(_r, _c, val)
            else:
                return True
            
        for r in range(rows):
            if not check(r, 0, matrix[r][0]):
                return False
            
        for c in range(1, cols):
            if not check(0, c, matrix[0][c]):
                return False
            
        return True
```


### [498. Diagonal Traverse](https://leetcode.com/problems/diagonal-traverse/)

```python
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        """
        Intuition: If mat[r1][c1] and mat[r2][c2] are in same diagnal,
        r1 + c1 == r2 + c2.
        Reverse a diagnal by the idx of the diagnal
        """
        rows, cols = len(mat), len(mat[0])
        mp = defaultdict(list)
        for r in range(rows):
            for c in range(cols):
                mp[r + c].append(mat[r][c])
        
        ans = []
        for i in range(rows + cols):
            if i % 2:
                ans.extend(mp[i])
            else:
                ans.extend(reversed(mp[i]))
        return ans
```