## Description

All 2D DP are actually grid problem. E.g. 1289. Minimum Falling Path Sum II vs. 265. Paint House II



## Square on Grid

### [221. Maximal Square](https://leetcode.com/problems/maximal-square/)
```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        dp[0][0] = 1 if matrix[0][0] == 1 else 0
        
        ans = 0
        
        for i in range(len(dp)):
            for j in range(len(dp[0])):
                if i == 0 or j == 0:
                    print(matrix[i][j])
                    dp[i][j] = 1 if matrix[i][j] == '1' else 0
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1 if matrix[i][j] == '1' else 0
                ans = max(ans, dp[i][j])
            
        return ans * ans
```
### [1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/)

```python
class Solution:
    def countSquares(self, M: List[List[int]]) -> int:
        R, C = len(M), len(M[0])
        
        # Intuition: Travese grid, update cell with the len of side of the squre
        # with the current cell as the bottom-right corner, the num is also the
        # num of squre with the cell as bottom-right corner.
        # 
        # See https://leetcode.com/problems/maximal-square/ to understand below
        # formula
        # M[r][c] = min(M[r - 1][c], M[r][c - 1], M[r - 1][c - 1]) + 1
        ans = 0
        for r in range(0, R):
            for c in range(0, C):
                if r > 0 and c > 0 and M[r][c]:
                    M[r][c] = min(M[r - 1][c], M[r][c - 1], M[r - 1][c - 1]) + 1
                ans += M[r][c]
        return ans
```

### [304. Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        R, C = len(matrix), len(matrix[0])
        dp = [[0] * C for _ in range(R)]
        dp[0][0] = matrix[0][0]
        
        for r in range(1, R):
            dp[r][0] = dp[r - 1][0] + matrix[r][0]
        
        for c in range(1, C):
            dp[0][c] = dp[0][c - 1] + matrix[0][c]
        
        for r in range(1, R):
            for c in range(1, C):
                dp[r][c] = dp[r][c - 1] + dp[r - 1][c] - dp[r - 1][c - 1] + matrix[r][c]
        
        self.dp = dp

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        # ans = self.dp[row2][col2] - self.dp[row1 - 1][col2] - 
        #       self.dp[row2][col1 - 1] + self.dp[row1 - 1][col1 - 1]
        ans = self.dp[row2][col2]
        if row1 > 0:
            ans -= self.dp[row1 - 1][col2]
        if col1 > 0:
            ans -= self.dp[row2][col1 - 1]
        if row1 > 0 and col1 > 0:
            ans += self.dp[row1 - 1][col1 - 1]
        
        return ans
```

## Path Sum

### [64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        r, c = len(grid), len(grid[0])
        dp = [[grid[0][0] for _ in range(c)] for _ in range(r)]
        for i in range(1, r):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1, c):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        for i in range(1, r):
            for j in range(1, c):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

### [931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/)

```python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        for r in range(1, R):
            for c in range(C):
                # Trick: Gate the border using min/max
                matrix[r][c] += min(matrix[r - 1][max(0, c - 1):min(C, c + 2)])
        return min(matrix[-1])
```

## Others


### [62. Unique Paths](https://leetcode.com/problems/unique-paths/)

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]
        for row in reversed(range(m - 1)):
            for col in reversed(range(n - 1)):
                dp[row][col] = dp[row + 1][col] + dp[row][col + 1]
        return dp[0][0]
```

### [688. Knight Probability in Chessboard](https://leetcode.com/problems/knight-probability-in-chessboard/)

```python
class Solution:
    def knightProbability(self, n: int, K: int, row: int, column: int) -> float:
        """
        Intuition: DP[k][r][c] is the possiblity of knight to be on (r, c) after k
        move.
        Use k as time machine, every cell has different possibiity in different time
        layer identified by k.
        """
        dp = [[[0] * n for _ in range(n)] for _ in range(K + 1)]
        dp[0][row][column] = 1
        for k in range(1, K + 1):
            for r in range(n):
                for c in range(n):
                    for _r, _c in ((r + 2, c + 1), (r - 2, c + 1), (r + 2, c - 1), \
                                   (r - 2, c - 1), (r + 1, c + 2), (r - 1, c + 2), \
                                   (r + 1, c - 2), (r - 1, c - 2)):
                        if 0 <= _r < n and 0 <= _c < n:
                            # Think reversly, dp[k][r][c]'s value comes from its source
                            # there are at most 8 sources, each valid source add possibilities
                            # to current position. Note /8.0
                            dp[k][r][c] += dp[k - 1][_r][_c] / 8.0
        
        # Trick: Sum of grid
        return sum(map(sum, dp[-1]))
    
    def knightProbability1(self, N: int, K: int, r: int, c: int) -> float:
        # Trick: Rolling array - state compression
        dp = [[0] * N for _ in range(N)]
        dp[r][c] = 1
        for k in range(K):
            # Each step, create new field
            dp2 = [[0] * N for _ in range(N)]
            for r in range(N):
                for c in range(N):
                    for dr, dc in ((1, 2), (-1, 2), (1, -2), (-1, -2),
                                  (2, 1), (-2, 1), (2, -1), (-2, -1)):
                        _r, _c = r + dr, c + dc
                        if 0 <= _r < N and 0 <= _c < N:
                            dp2[_r][_c] += dp[r][c] / 8.0
            dp = dp2
        return sum(map(sum, dp))
```

### [935. Knight Dialer](https://leetcode.com/problems/knight-dialer/)

```python
class Solution:
    def knightDialer(self, n: int) -> int:
        MOD = 10 ** 9 + 7
        key_map = {
            1: (6, 8),
            2: (7, 9),
            3: (4, 8),
            4: (3, 9, 0),
            5: (),
            6: (1, 7, 0),
            7: (2, 6),
            8: (1, 3),
            9: (2, 4),
            0: (4, 6)
        }
        # dp(i, n) is num of dials with n digits starting
        # from i
        @functools.lru_cache(None)
        def dp(i, n):
            if n == 1:
                return 1
            ans = 0
            for next_num in key_map[i]:
                ans += dp(next_num, n - 1)
                ans %= MOD
                
            return ans
        
        ans = 0
        for i in range(10):
            ans += dp(i, n)
        # ans = 4 * dp(1, n) + 2 * dp(2, n) + 2 * dp(4, n) + dp(0, n) + dp(5, n)
            
        return ans % MOD
```


### [63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if not obstacleGrid: return 0
        R, C = len(obstacleGrid), len(obstacleGrid[0])
        if not C or obstacleGrid[0][0] == 1: return 0
        dp = [[0] * C for _ in range(R)]
        dp[0][0] = 1
        dirs = ((0, -1), (-1, 0))
        for r in range(R):
            for c in range(C):
                if obstacleGrid[r][c] == 1:
                    continue
                for dr, dc in dirs:
                    _r, _c = r + dr, c + dc
                    if 0 <= _r < R and 0 <= _c < C and obstacleGrid[_r][_c] != 1:
                        dp[r][c] += dp[_r][_c]
        
        return dp[-1][-1]
    
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # Inuition: Update input in place to improve space
        if not obstacleGrid: return 0
        R, C = len(obstacleGrid), len(obstacleGrid[0])
        if not C or obstacleGrid[0][0] == 1: return 0
        
        # The tricky part is to handle [0][0]
        # Anothe tricky part is to handle the obstacle cell
        # - When the obstacle cell is cur cell, it is set to 0
        # - When the obstacle cell is pre cell used by cur, it is 0
        obstacleGrid[0][0] = 1
        for c in range(1, C):
            obstacleGrid[0][c] = obstacleGrid[0][c - 1] if not obstacleGrid[0][c] else 0
        for r in range(1, R):
            obstacleGrid[r][0] = obstacleGrid[r - 1][0] if not obstacleGrid[r][0] else 0
        
        for r in range(1, R):
            for c in range(1, C):
                if obstacleGrid[r][c]:
                    obstacleGrid[r][c] = 0
                else:
                    obstacleGrid[r][c] = obstacleGrid[r][c - 1] + obstacleGrid[r - 1][c]
                            
        return obstacleGrid[-1][-1]
```



### [576. Out of Boundary Paths](https://leetcode.com/problems/out-of-boundary-paths/)

```python
class Solution:
    def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
        @functools.lru_cache(None)
        def dp(i, j, k):
            # Intuition: dp(i, j, k) denotes max ways starting from (i, j) to move ball out boarder
            # with max k steps
            if i == m or j == n or i < 0 or j < 0:
                return 1
            # If no steps remains and the ball is still not out, it cannot work
            if k == 0:
                return 0
            return dp(i - 1, j, k - 1) + dp(i + 1, j, k - 1) + dp(i, j + 1, k - 1) + dp(i, j - 1, k - 1)
        return dp(i, j, N) % (10 ** 9 + 7)
```

### [174. Dungeon Game](https://leetcode.com/problems/dungeon-game/)

```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        rows, cols = len(dungeon), len(dungeon[0])
        # dp[i][j] is the min health the hero should have to survive BEFORE he
        # enter dungeon[i][j]
        dp = [[math.inf] * cols for _ in range(rows)]
        
        def get_health(row, col):
            if row >= rows or col >= cols:
                return math.inf
            return dp[row][col]
        
        # Trick: Start from right-bottom
        for row in reversed(range(rows)):
            for col in reversed(range(cols)):
                cur_value = dungeon[row][col]
                right_health = get_health(row, col + 1)
                down_health = get_health(row + 1, col)
                min_next_health = min(right_health, down_health)
                
                if min_next_health == math.inf:
                    # If it is first point at right-bottom, the health
                    min_cur_health = 1 if cur_value >= 0 else 1 - cur_value
                else:
                    # If has next room, the hero should have to enough health
                    # for next room after deal with cur room
                    min_cur_health = max(1, min_next_health - cur_value)
                dp[row][col] = min_cur_health
                    
        return dp[0][0]
```

### [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)