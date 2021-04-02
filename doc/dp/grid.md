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
    def knightProbability(self, N: int, K: int, r: int, c: int) -> float:
        # Inuition: dp[k][r][c] means after k steps, the posibility of knight to be
        # on point (r, c)
        dp = [[[0] * N for _ in range(N)] for _ in range(K + 1)]
        # Initial value: At step 0, posibility of (r, c) is 1
        dp[0][r][c] = 1
        for k in range(1, K + 1):
            for r in range(N):
                for c in range(N):
                    for dr, dc in ((1, 2), (-1, 2), (1, -2), (-1, -2),
                                  (2, 1), (-2, 1), (2, -1), (-2, -1)):
                        _r, _c = r + dr, c + dc
                        if 0 <= _r < N and 0 <= _c < N:
                            # Update 8 possible locations for each location
                            # at pre step, the possiblity is a accumulated value
                            # if there are duplicated moves from diff locations
                            # from pre step
                            dp[k][_r][_c] += dp[k - 1][r][c] / 8.0
        
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

