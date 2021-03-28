[TOC]
https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns
https://leetcode.com/problems/paint-house/solution/
Questions to print all ways cannot resolved using DP

Take unknown as known

pattern: 576, 1269

## Problems

### [264. Ugly Number II](https://leetcode.com/problems/ugly-number-ii/)

```python
class Ugly:
    def __init__(self):
        # Trick: Not use self.x
        self.nums = nums = [1]
        # Trick: 3 Pointers
        i2 = i3 = i5 = 0
        for i in range(1, 1690):
            ugly = min(nums[i2] * 2, nums[i3] * 3, nums[i5] * 5)
            nums.append(ugly)
            
            if ugly == nums[i2] * 2:
                i2 += 1
            if ugly == nums[i3] * 3:
                i3 += 1
            if ugly == nums[i5] * 5:
                i5 += 1

class Solution:
    # Trick: Load on initiation
    u = Ugly()
    def nthUglyNumber(self, n: int) -> int:
        return self.u.nums[n - 1]
```

### [312. Burst Balloons](https://leetcode.com/problems/burst-balloons/)

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        n = len(nums)
        
        # dp[i][j] is the max coins if adding baloons back to 
        # between i and j, i and j are not included. The location of baloons
        # added to btw i,j is decided by nums. Spaces btw i,j is currently
        # empty.
        dp = [[0] * n for _ in range(n)]
        
        # The current n is after reframe, so n - 1 is the dummy right node,
        # we want to calculate left: 0 ~ n-2, because left has to be less than right
        # right should be left + 2 ~ n - 1, because the purpose of left/right is to 
        # define walls for real baloon location to add baloons.
        # When left is n - 2, no right avaialbe, and no realy baloon locations, so
        # the 2nd loop won't run. 
        # To calculate max coin btw left/right exclusive, interate lef+1~right-1,
        # and calculate if put baloon there, and plus max coin of fill the two sub
        # section.
        for left in range(n - 2, -1, -1):
            for right in range(left + 2, n):
                dp[left][right] = max(nums[left] * nums[i] * nums[right] + dp[left][i] + dp[i][right] for i in range(left+1, right))
                
        return dp[0][n - 1]
```

### [1531. String Compression II](1531. String Compression II)
```python
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        # Trick: Top down DP
        #   Use states from parent step can help make decisions
        @functools.lru_cache(None)
        def helper(start, last, last_len, deletes):
            # The helper function returns the min len of str starting from start, give
            # following status from parent steps
            # start - the cur start of the sub problem
            # last - last char that included in the running str
            # last_len - the len of the last char in the running str
            # deletes - number of deletes available
            if deletes < 0:
                return math.inf
            if start >= len(s):
                return 0
            if s[start] == last:
                # If cur start is same as last, just add it and not attempt the delete case
                # because the delete case for same char is tried when the first same char
                # appear in else branch
                increase = 1 if last_len in (1, 9, 99) else 0
                return increase + helper(start + 1, last, last_len + 1, deletes)
            else:
                # For case char changed, try two possible situation
                # If delete the cur start, keep moving cur start, last no change
                # deletes reduce by 1
                deleted_len = helper(start + 1, last, last_len, deletes - 1)
                # If kep the cur start, because it is new char, len incrase by 1
                kept_len = 1 + helper(start + 1, s[start], 1, deletes)
                return min(deleted_len, kept_len)
        return helper(0, '', 0, k)
```



### [741. Cherry Pickup](https://leetcode.com/problems/cherry-pickup/)

```python
# Fav
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        # Trick: Two times DP cannot work
        # Trick: Two points grid search
        N = len(grid)
        @functools.lru_cache(None)
        def dp(r1, c1, c2):
            # dp is the max cherry collected by two people started at
            # (r1, c1) and (r2, c2) until (N - 1, N - 1)
            # Trick: If starting from (0,0) and only down/right a step,
            #   for step t, r + c = t
            r2 = r1 + c1 - c2
            if r1 == N or c1 == N or r2 == N or c2 == N or \
            grid[r1][c1] == -1 or grid[r2][c2] == -1:
                return -math.inf
            elif r1 == c1 == N - 1:
                # If person 1 reach destination, person 2 is there too
                # and because only one person can pick same cherry
                # the max cherry of final step is 1 or 0
                return grid[r1][c1]
            else:
                ans = grid[r1][c1] + (c1 != c2) * grid[r2][c2]
                ans += max(dp(r1 + 1, c1, c2 + 1),
                           dp(r1, c1 + 1, c2 + 1),
                           dp(r1 + 1, c1, c2),
                           dp(r1, c1 + 1, c2))
                return ans
            
        return max(0, dp(0, 0, 0))
```







### [983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/)
```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        N = len(days)
        @functools.lru_cache(None)
        def dp(d):
            # dp is the min cost starting from day d in days, that day is not covered
            # by prev duration. At the day d not covered by pre duration, it can buy
            # pass 1 or 7 or 30, for each case, the cost is cost[i] + dp(d_i), d_ is is
            # d1, d7 and d30, which are the first uncovered day when buy pass 1/7/30 at
            # current day
            if d >= N:
                return 0
            
            d1, d7, d30 = d + 1, N, N
            for i in range(d, N - 1):
                if days[i] - days[d] <= 6 and days[i + 1] - days[d] > 6:
                    d7 = i + 1
                if days[i] - days[d] <= 29 and days[i + 1] - days[d] > 29:
                    d30 = i + 1
                    break
            return min(costs[0] + dp(d1), costs[1] + dp(d7), costs[2] + dp(d30))
            
        return dp(0)
    
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        # Trick: Use hashset for lookup
        days = {*days}
        @functools.lru_cache(None)
        def dp(d):
            if d > 365:
                return 0
            if d not in days:
                # If d not travel, it is next day
                return dp(d + 1)
            else:
                return min(costs[0] + dp(d + 1), costs[1] + dp(d + 7), costs[2] + dp(d + 30))
        # Because the day value provided in days starting from 1, so init value=1 and
        # the term condition is > 365
        return dp(1)
```



### [1240. Tiling a Rectangle with the Fewest Squares](https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares/)
```python
class Solution:
    def tilingRectangle(self, n: int, m: int) -> int:
        # dp(state) is the num of extra tiles needs to be added to cover
        # all empty area in current state. The state has width of m and height
        # of n, the filling starting from the ground using skyline style
        @functools.lru_cache(None)
        def dp(state):
            # Handle end condition, if no empty area, 0 needs to be added
            min_h = min(state)
            if min_h == n:
                return 0
            
            # Find leftmost lowest empty area as start point
            state = list(state)
            start = state.index(min_h)
            
            # Starting from start, try all possibilities to cover using growing
            # size of tile until the height is different than start.
            # Get the min ans for the remaining area when covered using one size,
            # then the final return will be ans + 1
            ans = m * n
            for end in range(start, m):
                # Here if you use start[start] instead of min_h, it will be wrong,
                # because state changed all the time while we iterate all possible end.
                if state[end] != min_h:
                    break
                side = end - start + 1
                height = min_h + side
                if height > n:
                    break
                else:
                    state[start:end + 1] = [height] * side
                    ans = min(ans, dp(tuple(state)))
            return ans + 1
        
        # Trick: Make sure n > m
        # This is not required for this problem
        if m > n:
            m, n = n, m
        
        # Trick: When use lru_cache, cannot use list which is not hashable
        return dp(tuple([0] * m))
```

### [1269. Number of Ways to Stay in the Same Place After Some Steps](https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

```python
#   | 0  1
# --------
# 0 | 1  0
# 1 | 1  1
# 2 | 2  2
# 3 | 4  4
# 4 | 8  8

class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        # Pattern: For DP questions related to N steps on a grid or array,
        #   consider using steps as first demension then [i] or [i][j] as
        #   2nd and 3rd.
        # Tuition: No need to calculate all items in array, since pointer
        #   needs to go back to 0 in steps the fathest it can go is steps // 2,
        #   so the size of array is just (steps // 2 + 1);
        #   We can remodel the question to use an array of (steps // 2 + 1).
        #   Some math method should be used to prove the result are same, but
        #   I don't know. N actually can be any value between min(steps // 2 + 1, arrLen)
        
        N, MOD = min(steps // 2 + 1, arrLen), 10 ** 9 + 7
        dp = [[0] * (N) for _ in range(steps + 1)]
        dp[0][0] = 1
        
        for step in range(1, steps + 1):
            for i in range(N):
                if i == 0:
                    dp[step][i] = (dp[step - 1][i] + dp[step - 1][i + 1]) % MOD
                elif i == N - 1:
                    dp[step][i] = (dp[step - 1][i] + dp[step - 1][i - 1]) % MOD
                else:
                    dp[step][i] = (dp[step - 1][i - 1] + dp[step - 1][i] + dp[step - 1][i + 1]) % MOD
        
        return (dp[steps][0]) % MOD
```

### [1220. Count Vowels Permutation](https://leetcode.com/problems/count-vowels-permutation/)

```python
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        # dp[i][j] is when len of str is i + 1, starting from
        # vowel j, the count of strs
        dp = [[0] * 5 for _ in range(n)]
        dp[0] = [1] * 5
        MOD = 10 ** 9 + 7
        
        for i in range(1, n):
            dp[i][0] = (dp[i - 1][1]) % MOD
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % MOD
            dp[i][2] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][3] + dp[i - 1][4]) % MOD
            dp[i][3] = (dp[i - 1][2] + dp[i - 1][4]) % MOD
            dp[i][4] = (dp[i - 1][0]) % MOD
            
        return sum(dp[-1]) % MOD
```

### [790. Domino and Tromino Tiling](https://leetcode.com/problems/domino-and-tromino-tiling/)

```python
class Solution:
    def numTilings(self, N: int) -> int:
        # https://youtu.be/S-fUTfqrdq8
        # Inuition:
        # dp[i][0] - ways to cover i columns and last (i-th) colum are covered
        # dp[i][1] - ways to cover i columns and only first row of last column is covered
        # dp[i][2] - ways to cover i columns and only second row of last column is covered
        # dp[i][1] = dp[i][2], so only use one of them
        MOD = 10 ** 9 + 7
        dp = [[0] * 2 for _ in range(N + 1)]
        dp[0][0] = dp[1][0] = 1
        for i in range(2, N + 1):
            dp[i][0] = (dp[i - 1][0] + dp[i - 2][0] + 2 * dp[i - 1][1]) % MOD
            dp[i][1] = (dp[i - 2][0] + dp[i - 1][1]) % MOD
        return dp[-1][0]
```

### [808. Soup Servings](https://leetcode.com/problems/soup-servings/)

```python
class Solution:
    def soupServings(self, N: int) -> float:
        if N > 4800:
            return 1
        @functools.lru_cache(None)
        def dp(a, b):
            if a <= 0 and b <= 0: return 0.5
            if a <= 0: return 1
            if b <= 0: return 0
            return 0.25 * (dp(a - 100, b) + dp(a - 75, b - 25) +
                           dp(a - 50, b - 50) + dp(a - 25, b - 75))

        return dp(N, N)
```


