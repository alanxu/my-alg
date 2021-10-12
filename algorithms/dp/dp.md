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

### [1531. String Compression II](https://leetcode.com/problems/string-compression-ii/)
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

### [818. Race Car](https://leetcode.com/problems/race-car/)

```python
class Solution:
    # https://youtu.be/HzlEkUt2TYs
    dp = {0: 0}
    def racecar(self, t):
        if t in self.dp:
            return self.dp[t]
        n = t.bit_length()
        if 2**n - 1 == t:
            self.dp[t] = n
        else:
            self.dp[t] = self.racecar(2**n - 1 - t) + n + 1
            for m in range(n - 1):
                self.dp[t] = min(self.dp[t], self.racecar(t - 2**(n - 1) + 2**m) + n + m + 1)
        return self.dp[t]
```

### [837. New 21 Game](https://leetcode.com/problems/new-21-game/)

```python
'''
X, X, X, X, [i-w, i-w-1, ..., i-2, i-1], i
'''
class Solution:
    def new21Game(self, N: int, K: int, W: int) -> float:
        # Intuition: dp[i] is the possibility of getting i
        # points by keep drawing cards until getting >= K points.
        # For each point target dp[i], it can be obtained by
        # drawing 1 card after getting points [i -1, i - 2, ..., 
        # i - W], we know the probability of all those previous
        # situations, then it is 1.0/W probability to draw the card
        # required to make points i after all W previous cases.
        # So dp[i] = 1.0/W*dp[i - 1] + 1.0/W*dp[i - 2] + ... +
        # 1.0/W*dp[i - W].
        # Note that the previous cases are valid only if the points
        # < k, because if it is >=k, card drawing is stopped, there is 0 
        # probability to get to points i. So the base points taken in to
        # account to calc dp[i] should < k and maximun W points.
        dp = [0] * (N + 1)
        dp[0] = 1
        # Trick: Use sum to keep subarray sum
        sum_ = 0
        for i in range(1, N + 1):
            # Only invude prev points < K
            if i - 1 < K:
                sum_ += dp[i - 1]
            # If W + 1 points exists, remove it
            if i - W - 1 >= 0:
                sum_ -= dp[i - W - 1]
            dp[i] = 1.0 / W * sum_
            
        return sum(dp[K: N + 1])             
```

### [887. Super Egg Drop](https://leetcode.com/problems/super-egg-drop/)

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        @functools.lru_cache(None)
        def test(k, n):
            # Intuition: 
            # If k == 1, you can only
            # start from F1, when floor i break, you know the
            # answer is i - 1, so max move is n (worst case all
            # floors cannot break). So there is no case we cannot
            # get a number given k >= 1. 
            # If k > 1, the approach is, for given k and n, we choose a floor
            # i to start with, you first try 1 egg on F[i], if
            # break you need to test [1, i - 1] floors with k - 1
            # eggs, and if it doesn't break, you need to test
            # [i + 1, n] floors with k egg.
            # The tricky part is you don't know if it break or not,
            # you are just looking for worst case.
            
            # Why 0? The function always has a answer if k >= 1,
            # if k == 0  and n > 1, it means in upper recursion,
            # you have only 1 egg and you are trying it from bottom
            # of floor and at floor of upper recursion you assum
            # the only egg break, and the last egg is break, you know
            # the max moves (floor) for that case; so at cur resursion 
            # you get 0 egg you just return 0 so the upper recursion can
            # finish.
            if k == 0: return 0
            if k == 1: return n
            if n <= 1: return n
            
            ans = math.inf
            for i in range(1, n + 1):
                ans = min(ans, 1 + max(test(k, n - i), test(k - 1, i - 1)))
            
            return ans
        return test(k, n)
    
    def test(k, n):
        # Trick: Iterate for i in n get TLE, use binary search to find i.
        # The 2 funcs are monotone increasing and decreasing, so their
        # minmax is the point they meet... when test(k, n - i) == test(k - 1, i - 1)
        @functools.lru_cache(None)
        def test(k, n):
            if k == 0: return 0
            if k == 1: return n
            if n <= 1: return n
            
            l, r = 1, n
            while l < r:
                m = (l + r) // 2
                # When first try F[1], if break when got answer with 1 move,
                # and we need more move if it is not break. So when no_break > break
                # it means m needs to be increased by moving l
                if test(k, n - m) >= test(k - 1, m - 1):
                    l = m
                else:
                    r = m - 1
            
            return test(k, n - l)
        return test(k, n)
    
    def superEggDrop(self, K, N):
        # https://leetcode.com/problems/super-egg-drop/discuss/158974/C%2B%2BJavaPython-2D-and-1D-DP-O(KlogN)
        dp = [[0] * (K + 1) for i in range(N + 1)]
        for m in range(1, N + 1):
            for k in range(1, K + 1):
                dp[m][k] = dp[m - 1][k - 1] + dp[m - 1][k] + 1
            if dp[m][K] >= N: return m
```


### [1316. Distinct Echo Substrings](https://leetcode.com/problems/distinct-echo-substrings/)

```python
class Solution:
    def distinctEchoSubstrings(self, text: str) -> int:
        text = '?' + text
        N = len(text)
        
        # dp[i][j] denotes the length of same substring ending with i and j, j is after i
        dp = [[0] * N for _ in range(N)]
        words = set()
        ans = 0
        
        for i in range(1, N):
            for j in range(i + 1, N):
                # It is ok use N for upper border for both i and j
                # loop of j wont run when i == N - 1
                if text[i] == text[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                
                # Why >= not ==? Cuz the substr ending with j can be longer than
                # j - i
                if dp[i][j] >= j - i:
                    substr = text[i + 1:j + 1]
                    if substr not in words:
                        ans += 1
                        words.add(substr)
        
        return ans
```

### [91. Decode Ways](https://leetcode.com/problems/decode-ways/)

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        
        if not s or s[0] == 0:
            return 0
        
        dp = [0 for _ in range(len(s) + 1)]
        dp[0] = 1
        dp[1] = 0 if s[0] == '0' else 1
        
        for i in range(2, len(s) + 1):
            last_d = int(s[i - 2])
            cur_d = int(s[i - 1])
            
            n = last_d * 10 + cur_d
            
            if n == 0:
                return 0
            
            if cur_d == 0:
                if n <= 26:
                    dp[i] = dp[i - 2]
                else:
                    return 0
            elif last_d == 0 or n > 26:
                dp[i] = dp[i - 1]
            else:
                dp[i] = dp[i - 1] + dp[i - 2] 
                
        return dp[-1]
```

### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)

### [120. Triangle](https://leetcode.com/problems/triangle/)

```python
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        m = len(triangle)
        
        for i in reversed(range(0, m-1)):
            for j in range(len(triangle[i])):
                triangle[i][j] = min(triangle[i+1][j], triangle[i+1][j+1]) + triangle[i][j]
                
        print(triangle)
        
        return triangle[0][0]
```

### [1177. Can Make Palindrome from Substring](https://leetcode.com/problems/can-make-palindrome-from-substring/)

```python
class Solution(object):
    def canMakePaliQueries(self, s, queries):
        """
        :type s: str
        :type queries: List[List[int]]
        :rtype: List[bool]
        """
        # https://leetcode.com/problems/can-make-palindrome-from-substring/discuss/371999/Python-100-runtime-and-memory
        
        S = len(s)
        dp = [0] * (S + 1)
        a = ord('a')
        ints = map(lambda x: ord(x) - a, s)
        
        for i in range(1, S + 1):
            dp[i] = dp[i - 1] ^ (1 << ints[i - 1])
        
        ans = []
        for query in queries:
            left, right, k = query
            # Trick: Be carefull on edges when handling prefix difference
            diff = dp[left] ^ dp[right + 1]
            ans.append(bin(diff).count('1') <= (k * 2 + 1))
            
        return ans
```

### [1563. Stone Game V](https://leetcode.com/problems/stone-game-v/)
https://youtu.be/2_JkASlmxTA
```python
class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:
        N = len(stoneValue)
        dp = [[0] * N for _ in range(N)]
        
        for i in range(N - 1):
            dp[i][i + 1] = max(stoneValue[i], stoneValue[i + 1])
        
        for l in range(3, N):
            for i in range(0, N - l + 1):
                print(i)
                j = i + l - 1
                for k in range(i, j):
                    left_sum, right_sum = sum(stoneValue[i:k + 1]), sum(stoneValue[k + 1:j + 1])
                    if left_sum > right_sum:
                        dp[i][j] = max(dp[i][j], right_sum + dp[k + 1][j])
                    elif left_sum < right_sum:
                        dp[i][j] = max(dp[i][j], left_sum + dp[i][k])
                    else:
                        dp[i][j] = max(dp[i][j], left_sum + max(dp[k + 1][j], dp[i][k]))
        print(dp)
        return dp[0][-1]
    
    def stoneGameV(self, stoneValue: List[int]) -> int:
        n = len(stoneValue)
        acc = [0] + list(itertools.accumulate(stoneValue))

        @functools.lru_cache(None)
        def dfs(i, j):
            if j == i:
                return 0
            ans = 0
            for k in range(i, j):
                s1, s2 = acc[k + 1] - acc[i], acc[j + 1] - acc[k + 1]
                if s1 <= s2:
                    ans = max(ans, dfs(i, k) + s1)
                if s1 >= s2:
                    ans = max(ans, dfs(k + 1, j) + s2)
            return ans

        return dfs(0, n - 1)
    

    def stoneGameV(self, stoneValue: List[int]) -> int:
        n = len(stoneValue)
        # Trick: [0] make prefix sum operations easier
        acc = [0] + list(itertools.accumulate(stoneValue))

        @functools.lru_cache(None)
        def dfs(i, j):
            # Trick: j is exclusive, make logic easier
            if j - i == 1:
                return 0
            ans = 0
            for k in range(i + 1, j):
                s1, s2 = acc[k] - acc[i], acc[j] - acc[k]
                if s1 <= s2:
                    ans = max(ans, dfs(i, k) + s1)
                if s1 >= s2:
                    ans = max(ans, dfs(k, j) + s2)
            return ans

        return dfs(0, n)
```


### [877. Stone Game](https://leetcode.com/problems/stone-game/)

```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        N = len(piles)
        # Intuition: dp(i, j) is the max score Alice can get when remaining piles
        # are [i, j], WHEN both side play optimally. 
        @functools.lru_cache(None)
        def dp(i, j):
            if i > j:
                return 0
            side = (N - j + i - 1) // 2
            if side == 1:
                return max(piles[i] + dp(i + 1, j), piles[j] + dp(i, j - 1))
            else:
                return min(-piles[i] + dp(i + 1, j), -piles[j] + dp(i, j - 1))
        return dp(0, N - 1)
```

### [926. Flip String to Monotone Increasing](https://leetcode.com/problems/flip-string-to-monotone-increasing/)

```python
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        """
        Intuition:
        Solution 1: Greedy
        Use prefix to know hwo many 1's b/w any 2 elements,
        for each elements, supose it is begining of 1's, the flips is
        1's at its left plus 0's at its right
        
        Solution 2: DP
        dp is the ans for cur pos i. 
        If i is 1, dp[i] = dp[i - 1]
        If i is 0, dp[i] = min(flip i to 1, flip all 1's before i to 0)
        """
        
        dp = 0
        cur_ones, ones = 0, Counter(s)[1]
        
        for x in s:
            if x == '1':
                cur_ones += 1
            else:
                dp = min(dp + 1, cur_ones)
                
        return dp
```