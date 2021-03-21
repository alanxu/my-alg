
https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns
https://leetcode.com/problems/paint-house/solution/
Questions to print all ways cannot resolved using DP

Take unknown as known

pattern: 576, 1269

## Problems

### [139. Word Break](https://leetcode.com/problems/word-break/)
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        l = len(s)
        dp = [False] * (l + 1)
        dp[0] = True
        
        for i in range(1, l+1):
            for j in range(0, i):
                if dp[j] and s[j:i] in wordDict:
                    print(i)
                    print
                    dp[i] = True
                    break

        return dp[-1]
```

### [542. 01 Matrix](https://leetcode.com/problems/01-matrix/)
```python

```

### [1048. Longest String Chain](https://leetcode.com/problems/longest-string-chain/)
```python
class Solution:
    def longestStrChain1(self, words: List[str]) -> int:
        # Trick: DP - Bottom up
        # Trick: Use dict for DP, cus the key is str this time
        dp = {}
        # Trick: Sort the input
        # Words with same length for sure not in same chain
        for w in sorted(words, key=len):
            dp[w] = max([dp.get(w[:i] + w[i+1:], 0) + 1 for i in range(len(w))])
            
        return max(dp.values())
    
    def longestStrChain(self, words: List[str]) -> int:
        dp = {w: 0 for w in words}
        def dfs(w):
            if w not in dp:
                return 0
            
            # If dp[w] == 0, calculat it, otherwise use memory
            if not dp[w]:
                dp[w] = max(dfs(w[:i]+w[i+1:]) + 1 for i in range(len(w)))
                
            return dp[w]
        return max([dfs(w) for w in words])
```


### [1335. Minimum Difficulty of a Job Schedule](https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/)
```python
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        N = len(jobDifficulty)
        dp = [ [0] * (N + 1) for _ in range(d + 1)]
        dp[-1][-1] = math.inf
        ans = 0
        def backtrack(start, day):
            if day == d:
                return
            
            if day == d - 1:
                dp[day+1][N] = min(dp[day+1][N], dp[day][start] + max(jobDifficulty[start:]))
                # print((day, start, N-1, dp[day+1][N]))
                backtrack(N, day + 1)
            else:
                end = N - d + day
                for i in range(start, end + 1):
                    dp[day+1][i + 1] = dp[day][start] + max(jobDifficulty[start:i + 1])
                    # print((day, start, i, dp[day+1][i+1]))
                    backtrack(i + 1, day + 1)
        
        backtrack(0, 0)
        # print(dp)
        return dp[-1][-1] if N >= d else -1
    
    def minDifficulty(self, A: List[int], D: int) -> int:
        # https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/discuss/924611/DFS-greater-DP-Progression-with-Explanation-O(n3d)O(nd)
        N = len(A)
        if N < D:return -1
        # dp[i][j] is to partition jobs [0:j+1] withing j days
        dp = [[math.inf] * D for _ in range(N)]
        
        # Initiate value. put job 0 in day 0 only one possibility
        dp[0][0] = A[0]
        # For day0, the difficulty to hold i jobs are the max of
        # difficulty to hold i - 1 and A[i] it self
        for i in range(1, N):
            dp[i][0] = max(dp[i - 1][0], A[i])
            
        # Why range(1, N), A[0] must in dp[0][0]
        for i in range(1, N):
            # What does it mean, for i jobs, it can maximum fit into i days,
            # so, if i < d, it should be i days, as upper border so i + 1
            for j in range(1, min(i + 1, D)):
                # Given first i jobs, and first j days, trying to find min
                # difficulty dp[i][j], for jobs 1 ... i, say k jobs are paritioned
                # in j - 1 days, 1 <= k < i, then k + 1..i jobs in day j, and difficulty
                # of day j is is max(A[k + 1:i + 1]), difficulty of day 0...j is sum of
                # the two
                for k in range(i):
                    dp[i][j] = min(dp[i][j], dp[k][j - 1] + max(A[k + 1:i + 1]))
        # The result is min difficulty of N jobs partion in D days       
        return dp[-1][-1]
    
    
    def minDifficulty(self, A: List[int], D: int) -> int:
        N = len(A)
        if N < D: return -1
        
        # Trick: DFS
        #   i -> starting point of job, d -> days to be partitioned
        #   return: min job schedule difficulties
        @functools.lru_cache(None)
        def dfs(i, d):
            # If just put to 1 day, get the max of remaining jobs
            if d == 1:
                return max(A[i:])
            
            # Interate all possibilities within [i:N+1] jobs to put into
            # d days.
            # Why n - d + 1? There must be minimum jobs remaining for following
            # days. (n - 1) - (d - 1), then + 1 for upper boarder.
            # j is the ending job for cur day.
            res = math.inf
            for j in range(i, N - d + 1):
                res = min(res, max(A[i:j + 1]) + dfs(j + 1, d - 1))
            return res
        
        return dfs(0, D)
```

### [1235. Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/)

```python
# Fav
class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        N = len(startTime)
        # Trick: Bottom up DP
        # Trick: Sort jobs in start or end time
        #   Why sort?
        #     1. We need a support point to progress step by step,
        #        if sort by start, we can define the state dp[i] as max profit starting
        #        from job[i], because jobs are sorted by start time, job[i] value can be
        #        calculated following this definition;
        #        if sort by end, we can define the state dp[i] as max profit ending at job[i];
        #        we cannot define dp[i] as max profit starting from job[i] when sorted by end,
        #        because in this way job[i]'s profit cannot be calculated
        #     2. We need to check if two interval is overlapping or not
        #        when sorted, we can easily know if two intervals a, b is overlapping;
        #        to do that, typically we need to compare both
        #        - if b.start > a.start and b.start < a.end -> true
        #        - if a.start > b.start and a.start < b.end -> true
        #        by sorting a and b (e.g. by start time, a < b), we can just compare
        #        - if b.start < a.end -> true
        #        and you can only search array before/after job[i]
        # In this case, sorted by start time
        # Trick: Unpack a list
        start, end, profit = zip(*sorted(zip(startTime, endTime, profit)))
        
        # Trick: Find next no-overlapping job using bisect
        #   In jobs sorted by start, job[i]'s next job[j] when job[j].start >= job[i].end,
        #   next job is processed first, so the value is already there
        jump = {i: bisect.bisect_left(start, end[i]) for i in range(N)}
        
        # dp[i] is max profit starting from job i onwards
        # use N + 1 for edge case
        # dp[i] = max(profit_if_included, profit_if_excluded)
        # dp[i](profit_if_included) = profit[i] + dp[jump[i]]
        # dp[i](profit_if_excluded) = dp[i + 1] if not included, it is dp of next adjcent job
        dp = [0] * (N + 1)
        for i in range(N - 1, -1, -1):
            dp[i] = max(dp[i + 1], profit[i] + dp[jump[i]])
        
        # The result is the max profit starting from job[0]
        return dp[0]
    
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        # Trick: Top down DP (memory)
        n = len(startTime)
        start, end, profit = zip(*sorted(zip(startTime, endTime, profit)))
        jump = {i: bisect.bisect_left(start, end[i]) for i in range(n)}
        
        @functools.lru_cache()
        def max_profit_starting_from(i):
            if i == len(start):
                return 0
            return max(max_profit_starting_from(i + 1), profit[i] + max_profit_starting_from(jump[i]))
        
        return max_profit_starting_from(0)
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

### [140. Word Break II](https://leetcode.com/problems/word-break-ii/)

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        words = set(wordDict)
        @functools.lru_cache(None)
        def helper(s):
            if not s:
                return [[]]
            ans = []
            # end index is used to go through starting substr one char a step,
            # end is exclusive in the selected substr, so it starts at 1 to select
            # the first char as first starting word, and last possible case is the
            # whole s (last index=len(s) - 1), so end should be len(s), so in range
            # it shold be len(s) + 1, because range()'s end is inclusive
            for end in range(1, len(s) + 1):
                word = s[:end]
                if word in words:
                    for sublist in helper(s[end:]):
                        ans.append([word] + sublist)
            return ans
        
        return [' '.join(words) for words in helper(s)]
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

## Knapsack Problem

### [871. Minimum Number of Refueling Stops](https://leetcode.com/problems/minimum-number-of-refueling-stops/)

```python
class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        # Alg: Backpack - DP - Bottom Up
        #   Typical user case: Give a list of values, give some constrains for selection of values
        #   from the list, ask about the value of selected values (usually sum)
        #   Solution 1 is to use 2d DP, dp[i][j] where i alwasys means for first i items, j is the 
        #   most dynamic part related the constrains and question, dp is usually the sum
        
        # For this question, dp[i][j] is the max sum of fuels for first i stations when j stations 
        # are selected
        n = len(stations)
        # Size of dp is always n + 1 for the starting cases when i=0, j=0
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = startFuel
        # For first 1..n items
        for i in range(1, n + 1):
            station_idx = i - 1
            # Select 1..i from first i items
            for j in range(1, i + 1):
                # Not fuel
                dp[i][j] = dp[i - 1][j]
                
                # Fuel if possible
                if dp[i - 1][j - 1] >= stations[station_idx][0]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + stations[station_idx][1])
        
        # Given the clculated info, return first j when the cumulated fuel >= target
        for i in range(n + 1):
            if dp[-1][i] >= target:
                return i
        return -1
    
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        # Trick: State Compression
        #   Because dp[i] only use dp[i - 1], so can just same two dp rows,
        #   we can use rolling array, but we can even use one row, to do this we should
        #   start from end in 2nd loop
        n = len(stations)
        dp = [startFuel] + [0] * n
        for i in range(n):
            for j in range(i, -1, -1):
                if dp[j] >= stations[i][0]:
                    dp[j + 1] = max(dp[j + 1], dp[j] + stations[i][1])
        
        for i in range(n + 1):
            if dp[i] >= target: return i
        return -1        
```

### [416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum & 1: return False
        subset_sum = total_sum // 2
        n = len(nums)
        
        # dp[i][j] - If nums[0]...nums[i-1] can firm sum as j
        dp = [[False] * (subset_sum + 1) for _ in range(n + 1)]
        # It is true to firm sum as 0 by not selecting any num
        dp[0][0] = True
        
        for i in range(1, n + 1):
            cur = nums[i - 1]
            for j in range(subset_sum + 1):
                if j < cur:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - cur]
        return dp[-1][-1]
```

### [1049. Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/)

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        # Inuition: Easier way to think of the mim left of cancellation is to
        #   imagin seperate stones into 2 sets, if two sets have min diff, the
        #   diff is the min remaining. So the problem is to find max(sum) <= S / 2,
        #   where S is sum of all stones. So this is knapsack problem:
        #   Select some stones from stones, the max sum <= S / 2, calc the max
        #   value; W = S / 2, w[i] = stones[i], v[i] = stones[i]
        S, W, N = sum(stones), sum(stones) // 2, len(stones)
        dp = [[0] * (W + 1) for _ in range(N + 1)]
        
        for i in range(1, N + 1):
            for j in range(1, W + 1):
                # Not count
                dp[i][j] = dp[i - 1][j]
                
                # Count
                if j >= stones[i - 1]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j-stones[i - 1]] + stones[i - 1])
        
        # The diff is S minus both smashed parts
        return S - 2 * dp[-1][-1]
```

### [474. Ones and Zeroes](https://leetcode.com/problems/ones-and-zeroes/)
```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        S = [collections.Counter(s) for s in strs]
        S = [[c['0'], c['1']] for c in S]
        N = len(strs)
        
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(N + 1)]
        
        for i in range(1, N + 1):
            zeros, ones = S[i - 1]
            # The lower boarder of the constrains needs to be paid attention
            # sometimes start from 1 sometimes 0, it is up to the condition
            for j in range(0, m + 1):
                for k in range(0, n + 1):
                    dp[i][j][k] = dp[i - 1][j][k]
                    
                    if j >= zeros and k >= ones:
                        # Watch the i - 1
                        dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - zeros][k - ones] + 1)
        
        return dp[-1][-1][-1]
```

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

### [322. Coin Change](https://leetcode.com/problems/coin-change/)
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
                
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Intuition: Coin Change is a knapsack problem
        dp = [[math.inf] * (amount + 1) for _ in range(len(coins) + 1)]
        for i in range(len(coins) + 1):
            dp[i][0] = 0
        
        for i in range(1, len(coins) + 1):
            for j in range(1, amount + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= coins[i - 1]:
                    # Why it is dp[i][j - coins[i - 1]] not dp[i - 1][..]?
                    # because coins are not unique, not like backpack!!!
                    dp[i][j] = min(dp[i][j], dp[i][j - coins[i - 1]] + 1)
        return dp[-1][-1] if dp[-1][-1] != float('inf') else -1
```

### [1155. Number of Dice Rolls With Target Sum](https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/)

```python
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # Alg: Backpack - DP
        #   The unique feature of this question is that all items are
        #   selected, so need to count when not select current item
        dp = [[0] * (target + 1) for _ in range(d + 1)]
        dp[0][0] = 1
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                for k in range(1, min(j, f) + 1):
                    dp[i][j] += dp[i - 1][j - k]
        return dp[-1][-1] % (10 ** 9 + 7)
```

### [494. Target Sum](https://leetcode.com/problems/target-sum/)

```python
# Fav
# https://youtu.be/r6Wz4W1TbuI
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        
        # We just need last round so no need to keep all history
        dp = [0] * 2001
        dp[nums[0] + 1000] = 1
        dp[-nums[0] + 1000] += 1

        for i in range(1, len(nums)):
            # For each current round, create a new dp state
            cur = [0] * 2001
            for j in range(2001):
                ways = dp[j]
                if ways:
                    cur[j + nums[i]] += ways
                    cur[j - nums[i]] += ways
            # Update current state and dp to be used for next step
            dp = cur
                    
        return dp[S + 1000] if S <= 1000 else 0
    
    def findTargetSumWays5(self, nums: List[int], S: int) -> int:
        # dp is for first ith number, and each possible sum of i numbers, 
        # the number of ways to achieve that situation
        
        # So dp is a len(nums) * 2001 matrix
        # 2001 is max sum of all numbers with +/-, -1000 when all are -,
        # 1000 when all are +. Plus 1 for sum = 0.
        # Track: Use 1000 as offset to match array index (it has to be non-negative)
        dp = [[0] * 2001 for _ in range(len(nums))]
        # Set initial state, for first num, sum is +/-nums[0]
        # and both has 1 way.
        # Use +=1 for second one, cuz it nums[0] might be 0 leads to same sum
        dp[0][nums[0] + 1000] = 1
        dp[0][-nums[0] + 1000] += 1
        
        # Calculate every first i nums, 1 <= i < len(nums)
        for i in range(1, len(nums)):
            # For each possible sum of first i - 1 nums, if there are ways to make
            # a sum j, then increase ways to make sum (j +/- nums[i]) for first i nums
            # by j, not set to j, but increase by j, cuz other state of i - 1 will also
            # add to them
            for j in range(2001):
                ways = dp[i - 1][j]
                if ways:
                    dp[i][j + nums[i]] += ways
                    dp[i][j - nums[i]] += ways
                    
        # Final result is for all nums, when sum is S, the dp value
        return dp[-1][S + 1000] if S <= 1000 else 0
    
    def findTargetSumWays4(self, nums: List[int], S: int) -> int:
        # Without memo, the func calc() will be called 2 ^ n times, 
        # but max cases of sum is -1000 ~ 1000,
        # so there are a lot of i, cur_sum duplicated calculation.
        # Use memo to optimize it.
        memo = [[math.inf] * 2001 for _ in range(len(nums))]
        def calc(i=0, cur_sum=0):
            if i == len(nums):
                return 1 if cur_sum == S else 0
            else:
                if memo[i][cur_sum + 1000] != math.inf:
                    return memo[i][cur_sum + 1000]
                add = calc(i + 1, cur_sum + nums[i])
                subtract = calc(i + 1, cur_sum - nums[i])
                memo[i][cur_sum + 1000] = add + subtract
                return memo[i][cur_sum + 1000]
        return calc(0, 0)
    
    def findTargetSumWays3(self, nums: List[int], S: int) -> int:
        def calc(i=0, cur_sum=0):
            # Calculate num of ways starting from i to len(nums) - 1,
            # where acumulate sum of 0 ~ (i - 1) is cur_sum
            # and target final sum is S
            
            if i == len(nums):
                return 1 if cur_sum == S else 0
            else:
                add = calc(i + 1, cur_sum + nums[i])
                subtract = calc(i + 1, cur_sum - nums[i])
                return add + subtract
        return calc(0, 0)
    

    def findTargetSumWays2(self, nums: List[int], S: int) -> int:
        # This is got TLE, because no memory
        # Backtracking doesnt work with memory, because the input for
        # each recursion is different
        self.ans = 0
        def backtrack(symbols=[]):
            cur_idx = len(symbols)
            if cur_idx == len(nums):
                s = 0
                for i, n in enumerate(nums):
                    s = s + n if symbols[i] == '+' else s - n
                if s == S:
                    self.ans += 1
            else:
                for symbol in ('+', '-'):
                    symbols.append(symbol)
                    backtrack(symbols)
                    symbols.pop()
        backtrack()
        return self.ans
```

### [377. Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/)

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # Intuition: The problem is very much like coin change problem
        # https://leetcode.com/problems/coin-change/
        # there difference is this question asks for all possible possibilities
        # while coin change problem asks for min number for each target.
        #
        # There is a slight difference in the implementation, which is the order
        # of loops. In this project, we firstly loop target then loop on nums,
        # this make sure for each target starting from 1, we have computed the
        # complete num of combinations that can be used by following targets.
        #
        # While coin change just want the min value, so just focus on the latest
        # value.
        n = len(nums)
        dp = [0] * (target + 1)
        dp[0] = 1
        # Minor optimize: Sort the nums then break when num > j
        # nums.sort()
        for j in range(1, target + 1):
            for num in nums:
                if j >= num:
                    dp[j] = dp[j] + dp[j - num]
                # else:
                #     break
        return dp[-1]
```

## Max/min sums/costs

### [746. Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # Reframe the cost to add groud and top flow in the stairs
        cost = [0] + cost + [0]
        n = len(cost)
        # dp[i] is the min cost to get step i,
        # Initial state:
        # - It costs 0 to be on ground dp[0]
        # - It costs 0 to be on step 1, step 1's cost is cost[0], but it happen only
        #   when step 1 to step 2.
        dp = [0] * (n)
        dp[0] = dp[1] = 0
        
        for i in range(2, n):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
            
        return dp[-1]
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

### [650. 2 Keys Keyboard](https://leetcode.com/problems/2-keys-keyboard/)

```python
class Solution:
    def minSteps(self, n: int) -> int:
        if n == 1: return 0
        # dp[i] is min steps to produce i 'A'
        dp = [math.inf] * (n + 1)
        dp[1] = 0
        dp[2] = 2
        for i in range(3, n + 1):
            # j is the last index for first part
            for j in range(1, i):
                if (i - j) % j == 0:
                    dp[i] = min(dp[i], dp[j] + 1 + (i - j) // j)
                    
        return dp[-1]
```

### [279. Perfect Squares](https://leetcode.com/problems/perfect-squares/)

```python
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        square_nums = [i**2 for i in range(0, int(math.sqrt(n))+1)]
        
        dp = [float('inf')] * (n+1)
        # bottom case
        dp[0] = 0
        
        for i in range(1, n+1):
            for square in square_nums:
                if i < square:
                    break
                dp[i] = min(dp[i], dp[i-square] + 1)
        
        return dp[-1]
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


## Distinct Ways

### [70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1: return n
        dp = [0] * n
        dp[0], dp[1] = 1, 2
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]
```

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


## Paint House

### [1223. Dice Roll Simulation](https://leetcode.com/problems/dice-roll-simulation/)

```python
class Solution:
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        faces = 6
        # Intuition: Think about it is n dices in a row and only one roll
        # dp[i][j][k] is for first i dices, when last dice is j, there are k
        # duplication of j, the num of all possible cases.
        # There are max 15consecutive occurances for each face, so k <= 15,
        # to handle 0/1 case corner, we give treat k <= 16
        dp = [[[0] * (15 + 1) for _ in range(faces)] for _ in range(n)]
        # Initial cases: For first 1 dice, for each fase, only k=1 is valid, 
        # it only equals 1
        for i in range(faces):
            dp[0][i][1] = 1
        
        for i in range(1, n):
            for j in range(faces):
                # Don't forget +1, only calculate valid occurance
                for k in range(1, rollMax[j] + 1):
                    # If k == 1, the value equals sum of all first i - 1
                    # and last face is not j
                    if k == 1:
                        for jj in range(faces):
                            if jj != j:
                                for kk in range(1, rollMax[jj] + 1):
                                    dp[i][j][k] += dp[i - 1][jj][kk]
                    else:
                    # If k > 1, it is simple
                        dp[i][j][k] = dp[i - 1][j][k - 1]
        
        ans = 0
        for j in range(faces):
            ans += sum(dp[-1][j])
        
        # Trick: int() for 1e9
        return ans % int(1e9 + 7)
```

### [256. Paint House](https://leetcode.com/problems/paint-house/)

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        if not costs: return 0
        
        import copy
        dp = costs[-1]
        
        for i in range(len(costs) - 2, -1, -1):
            dp2 = copy.deepcopy(costs[i])
            dp2[0] += min(dp[1], dp[2])
            dp2[1] += min(dp[0], dp[2])
            dp2[2] += min(dp[0], dp[1])
            dp = dp2
        
        return min(dp)
```

### [265. Paint House II](https://leetcode.com/problems/paint-house-ii/)

```python
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        if not costs: return 0
        N, K = len(costs), len(costs[0])
        dp = [[0] * K for _ in range(N + 1)]
        
        for i in range(1, N + 1):
            for k in range(K):
                dp[i][k] = costs[i - 1][k] + min(dp[i - 1][:k] + dp[i - 1][k + 1:] or [0])
                
        return min(dp[-1])
```

### [276. Paint Fence](https://leetcode.com/problems/paint-fence/)

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return k
        dp = [0] * (n + 1)
        dp[1], dp[2] = k, k * k
        for i in range(3, n + 1):
            #https://leetcode.com/problems/paint-fence/discuss/178010/The-only-solution-you-need-to-read
            dp[i] = (dp[i - 1] + dp[i - 2]) * (k - 1)
        return dp[-1]
```


## String and Array

### [801. Minimum Swaps To Make Sequences Increasing](https://leetcode.com/problems/minimum-swaps-to-make-sequences-increasing/)

```python
class Solution:
    def minSwap(self, A: List[int], B: List[int]) -> int:
        # https://youtu.be/__yxFFRQAl8
        N = len(A)
        # Intuition: For each pos i, swap/keep denotes min swaps needed
        # to keep A, B valid with/without i-th num swapped
        swap, keep = [math.inf] * N, [math.inf] * N
        swap[0], keep[0] = 1, 0
        
        for i in range(1, N):
            # All the [i - 1] state are valid!
            # Based on comparasion of A[i], A[i - 1], B[i], B[i - 1],
            # there are at most 2 possiblilityes for swap and keep,
            # try to compute all of them and get the min
            if A[i] > A[i - 1] and B[i] > B[i - 1]:
                # This is best guess at the moment, when we don't know
                # relation betwwen A[i] and B[i - 1]
                swap[i] = swap[i - 1] + 1
                keep[i] = keep[i - 1]
            if A[i] > B[i - 1] and B[i] > A[i - 1]:
                swap[i] = min(swap[i], keep[i - 1] + 1)
                keep[i] = min(keep[i], swap[i - 1])
                
        return min(swap[-1], keep[-1])
```


### [673. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        N = len(nums)
        if N <= 1: return N
        
        lengths, counts = [0] * N, [1] * N
        
        for j in range(N):
            for i in range(j):
                if nums[j] > nums[i]:
                    if lengths[i] >= lengths[j]:
                        # Add a new element to existing incrasing seq, 
                        # len + 1, count remains same 
                        lengths[j] = lengths[i] + 1
                        counts[j] = counts[i]
                    elif lengths[i] + 1 == lengths[j]:
                        # Find another increasing sequence ending at i
                        # of same length equals cur max
                        counts[j] += counts[i]
        
        longest = max(lengths)
        return sum(c for i, c in enumerate(counts) if lengths[i] == longest)
```



## Others

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


## Divide and Conquer

### [1130. Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)

```python
class Solution:
    def mctFromLeafValues(self, A):
        # https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/discuss/339959/One-Pass-O(N)-Time-and-Space
        ans = 0
        while len(A) > 1:
            i = A.index(min(A))
            # Trick: why use substring sum? - avoid edge cases
            ans += min(A[i - 1:i] + A[i + 1:i + 2]) * A.pop(i)
        return ans
    
    def mctFromLeafValues(self, A):
        N = len(A)
        # dp[i][j] is the answer for substr [i][j]
        dp = [[math.inf] * N for _ in range(N)]
        
        # Initial value: single value substr yield 0
        for i in range(N):
            dp[i][i] = 0
        
        # Trick: Cannot simple use 2 loop for i and j, in that way not work
        #   First process all the sub str starting from 2, when len is 2
        #   and partition by k, the substr len is 1 which has value 0, then
        #   all len 3 has partion of len 1 or 2 which also calculated.
        for l in range(2, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                # 0 or 2 children
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], 
                                  max(A[i:k + 1]) * max(A[k + 1:j + 1]) + dp[i][k] + dp[k + 1][j])

        return dp[0][N - 1]
    
    def mctFromLeafValues(self, A):
        @functools.lru_cache(None)
        def dp(i, j):
            if i == j: return 0
            ans = math.inf
            for k in range(i, j):
                ans = min(ans, max(A[i:k+1]) * max(A[k+1:j+1]) + dp(i, k) + dp(k + 1, j))
            return ans
        return dp(0, len(A) - 1)
```

### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
                
        return dp[-1]
```


### [1039. Minimum Score Triangulation of Polygon](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/)

```python
#
#    |  0  1  2  3
#  --------------------
#  0 |  0  0  84 
#  1 |     0  0  140
#  2 |        0  0 
#  3 |  0        0
# 
#

class Solution(object):
    def minScoreTriangulation(self, A):
        # Pattern: Partition
        # For each subarray i~j, pick k in btw, i and j are disconnected
        N = len(A)
        dp = [[0] * N for _ in range(N)]
        
        # The length is num of edge not 
        # num of nodes! So it is 2 ~ (N - 1),
        # this makes the -/+ l clean
        for l in range(2, N):
            for i in range(N - l):
                j = i + l
                # For each k btw i and j exclusive
                dp[i][j] = math.inf
                for k in range(i + 1, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + A[i] * A[k] * A[j])
        
        return dp[0][-1]
```


### [546. Remove Boxes](https://leetcode.com/problems/remove-boxes/)

```python
class Solution:
    # https://leetcode.com/problems/remove-boxes/discuss/101310/Java-top-down-and-bottom-up-DP-solutions
    def removeBoxes(self, boxes: List[int]) -> int:
        N = len(boxes)
        dp = [[[0] * N for _ in range(N)] for _ in range(N)]
        # The initial states for k are based on assumption,
        # during iteration to populate true value, only the real
        # case will be visited
        for i in range(N):
            # boxes[i] is the (i+1)-th box, so there is i
            # possible boxes to the left of boxes[i], so the loop
            # should include i for k
            for k in range(i + 1):
                dp[i][i][k] = (k + 1) * (k + 1)
        
        # Trick: If l starts from 1 to N - 1, the code is cleaner
        # Trick: Interate from smaller len, so DP can work
        for l in range(1, N):
            for j in range(l, N):
                i = j - l
                for k in range(i + 1):
                    # If remove boxes[i]
                    dp[i][j][k] = (k + 1) * (k + 1) + dp[i + 1][j][0]
                    # If keep boxes[i]
                    # Try all values for i < m <= j where i and m same color
                    # dp[i][j] = (max value if delete all btw i and m exclusive) + 
                    # (max value of delete boxes[i] + boxes[m, j+1])
                    for m in range(i + 1, j + 1):
                        if boxes[m] == boxes[i]:
                            dp[i][j][k] = max(dp[i][j][k], dp[i + 1][m - 1][0] + dp[m][j][k + 1])

        return dp[0][N -1][0]
    
    def removeBoxes(self, boxes: List[int]) -> int:
        @functools.lru_cache(None)
        def dp(i, j, k):
            if i > j:
                return 0

            # prune the calculation
            while i + 1 <= j and boxes[i + 1] == boxes[i]:
                i += 1
                k += 1

            res = (k + 1) * (k + 1) + dp(i + 1, j, 0)
            for t in range(i + 1, j + 1):
                if boxes[t] == boxes[i]:
                    res = max(res, dp(i + 1, t - 1, 0) + dp(t, j, k + 1))
            return res

        return dp(0, len(boxes) - 1, 0)
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

### [1000. Minimum Cost to Merge Stones](https://leetcode.com/problems/minimum-cost-to-merge-stones/)

```python
class Solution:
    def mergeStones(self, stones: List[int], K: int) -> int:
        N = len(stones)
        if (N - 1) % (K - 1): return -1
        dp = [[0] * N for _ in range(N)]
        
        for l in range(K, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                dp[i][j] = math.inf
                for k in range(i, j, K - 1):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j])
                if (l - 1) % (K - 1) == 0:
                    dp[i][j] += sum(stones[i:j + 1])
        
        return dp[0][-1]
    
    def mergeStones1(self, stones: List[int], K: int) -> int:
        N = len(stones)
        if (N - 1) % (K - 1): return -1
        prefix = [0] * (N+1)
        for i in range(1,N+1): prefix[i] = stones[i-1] + prefix[i-1]
        dp = [[0] * N for _ in range(N)]
        for m in range(K, N+1):
            for i in range(N-m+1):
                dp[i][i+m-1] = min(dp[i][k] + dp[k+1][i+m-1] for k in range(i, i+m-1, K-1)) + (prefix[i+m] - prefix[i] if (m-1)%(K-1) == 0 else 0)
        return dp[0][N-1]
```

### [375. Guess Number Higher or Lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/)

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # Pattern: DP - partition
        # Intuition: Given a number nn, we have to find the worst case cost of 
        #   guessing a number chosen from the range (1, n)(1,n), assuming that 
        #   the guesses are made intelligently(minimize the total cost).
        #   dp[i][j] is the answer for substr [i, j] both inclusive,
        #   the maxmin cost is the cost assume you find the correct num at
        #   the very last. So if their is only one num, cost is 0, cuz it has
        #   to be left, if more than 1, first pick one (wrong) number other than
        #   last one, split to 3 parts. For len = 2, pick the first smaller num.
        #   
        #   The size of dp is (n + 1), because for dp[i][k - 1] when i = 1, if i start
        #   from 0, it become dp[0][-1], not correct
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for l in range(2, n + 1):
            for i in range(1, n - l + 1 + 1):
                j = i + l - 1
                dp[i][j] = math.inf
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], k + max(dp[i][k - 1], dp[k + 1][j]))
        
        return dp[1][n]
```


## DP on String

### [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

```python
class Solution:
    # Pattern: DP - String Matching
    #   use dp[i][j] to match each location i, j in s1 and s2
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        M, N = len(text1), len(text2)
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                    
        return dp[-1][-1]
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        M, N = len(text1), len(text2)
        # dp is for a i in text1, the answer for text1[i] and text2[:] 
        dp = [0] * (N + 1)
        
        for i in range(1, M + 1):
            dp2 = [0] * (N + 1)
            for j in range(1, N + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp2[j] = dp[j - 1] + 1
                else:
                    dp2[j] = max(dp2[j - 1], dp[j])
            dp = dp2
                
        return dp[-1]
```

### [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # Pattern: DP - On a single string
        N, ans = len(s), 0
        dp = [[False] * N for _ in range(N)]
        
        for i in range(N):
            dp[i][i] = True
            ans += 1
            if i < N - 1:
                dp[i][i + 1] = (s[i] == s[i + 1])
                ans += dp[i][i + 1]
        
        # Use len as the support point
        for l in range(3, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                # Trick: Check parlindrom of shorter substr
                dp[i][j] = dp[i + 1][j - 1] and (s[i] == s[j])
                ans += dp[i][j]
        
        return ans
```

### [516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # Pattern: Another way other than using l 
        # https://leetcode.com/problems/longest-palindromic-subsequence/discuss/99101/Straight-forward-Java-DP-solution
        N = len(s)
        dp = [[1] * N for _ in range(N)]
        
        for i in range(N - 1):
            dp[i][i + 1] = 1 + (s[i] == s[i + 1])
            
        for l in range(3, N + 1):
            for i in range(0, N - l + 1):
                j = i + l - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][-1]
```

### [1092. Shortest Common Supersequence](https://leetcode.com/problems/shortest-common-supersequence/)

```python
class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        # Pattern: DP - Longest Common Subsequence
        M, N = len(str1), len(str2)
        dp = [[''] * (N + 1) for _ in range(M + 1)]
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + str1[i - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
        
        lcs = dp[-1][-1]
        
        # Trick: Construct Shortest Common Superseq from LCS
        ans, i, j = "", 0, 0
        for c in lcs:
            while str1[i] != c:
                ans += str1[i]
                i += 1
            while str2[j] != c:
                ans += str2[j]
                j += 1
            ans += c
            i, j = i + 1, j + 1
            
        return ans + str1[i:] + str2[j:]
```

### [72. Edit Distance](https://leetcode.com/problems/edit-distance/)

```python
class Solution:
    # https://medium.com/@ethannam/understanding-the-levenshtein-distance-equation-for-beginners-c4285a5604f0
    def minDistance(self, word1: str, word2: str) -> int:
        M, N = len(word1), len(word2)
        if M * N == 0: return M + N
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        for i in range(M + 1):
            dp[i][0] = i
        for j in range(N + 1):
            dp[0][j] = j
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # min of below possibilities
                    # replace i: dp[i - 1][j - 1] + 1
                    # delete i: dp[i - 1][j] + 1
                    # insert i: dp[i][j - 1] + 1
                    # (somehow make first i, and j - 1 same, then
                    # add same w2[j] in w1[i] position)
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]
```

### [115. Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # Pattern: DP - String Matching
        # 1. Demention of dp is 1 + len()
        # 2. Loop for i, j start from 1 end len + 1
        # 3. s[i - 1] not s[i]
        M, N = len(s), len(t)
        if not M: return 0
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        # Initial value: for every source, if target is empty,
        # dp is 1
        for i in range(M + 1):
            dp[i][0] = 1
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                # dp[i][j] must include max num of subseq in dp[i - 1][j]
                # same target, pre source withouth i-th
                dp[i][j] = dp[i - 1][j]
                
                # If the char matches, all max num of pre target in pre
                # source counts
                if s[i - 1] == t[j - 1]:
                    dp[i][j] += dp[i - 1][j - 1]
        
        return dp[-1][-1]
```

### [712. Minimum ASCII Delete Sum for Two Strings](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/)

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        M, N = len(s1), len(s2)
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        # Initate all states where one of s is empty
        for i in range(1, M + 1):
            dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        for j in range(1, N + 1):
            dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
        
        # Start from first char of both
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # You know cost of make [i - 1] == [j], then
                    # cost of make [i] = [j] is just cost of[i - 1]
                    # plus cost ord([i])
                    dp[i][j] = min(dp[i - 1][j] + ord(s1[i - 1]), 
                                   dp[i][j - 1] + ord(s2[j - 1]))
        return dp[-1][-1]
```

### [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

```python
class Solution:
    
    def longestPalindrome(self, s: str) -> str:
        def expand(i):
            centers = [(i, i), (i, i + 1)]
            max_l = (0, -1, -1)
            
            for c in centers:
                j, k = c[0], c[1]
                while j >=0 and k < len(s) and s[j] == s[k]:
                    j -= 1
                    k += 1
                # -1 because j and k is 2 unit longer than the valid len
                if k - j - 1 > max_l[0]:
                    max_l = (k - j -1, j + 1, k -1)
            return max_l
                
            
        max_l = (0, -1, -1)
        for i in range(len(s)):
            l, s_, e = expand(i)
            if l > max_l[0]:
                max_l = (l, s_, e)
        
        print(max_l)
        return s[max_l[1]:max_l[2] + 1]
    
    def longestPalindrome(self, s: str) -> str:
        N = len(s)
        if N in (0, 1): return s
        dp = [[0] * N for _ in range(N)]
        # If no further updates, this is the answer
        ans, ans_len = s[0], 1
        
        for i in range(N):
            dp[i][i] = True
            if i < N - 1:
                dp[i][i + 1] = (s[i] == s[i + 1])
                if dp[i][i + 1] and ans_len == 1:
                    ans = s[i:i + 2]
                    ans_len = 2
        
        for l in range(3, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                dp[i][j] = dp[i + 1][j - 1] and s[i] == s[j]
                if dp[i][j] and l > ans_len:
                    ans = s[i:j + 1]
                    ans_len = l
        
        return ans
```

## Classic

### [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

```python
class Solution(object):
    def maxProfit(self, k, prices):
        """
        mp[kk][nn]: Max profit of kth transaction
        """
        
        n = len(prices) 
        
        if n <= 0:
            return 0
        
        mp = [[0 for _ in range(n)] for _ in range(k+1)]
        mp_result = 0
        
        for kk in range(1, k+1):
            # - prices[nn], because later on needs to caculate new max for kk by prices[nn] - prices[tt]
            temp_max = mp[kk-1][0] - prices[0]
            for nn in range(1, n):
                mp[kk][nn] = max(mp[kk][nn-1], temp_max + prices[nn])
                temp_max = max(temp_max, mp[kk-1][nn] - prices[nn])
                mp_result = max(mp[kk][nn], mp_result)
        
        return mp_result
    
    def maxProfit(self, K, prices):
        # Intuition:
        # 1. 1 transaction means buy AND sell once
        # 2. ans is always when no stock hold, because dp[k][n][1] < dp[k - 1][n - 1][0]
        # 3. dp[k][i][j] denotes on k-th transaction, on i-th day, when there are j stock 
        # (j is 0 or 1, 0 means no stock held k-th tran is done, 1 means 1 stock is bought
        # at k-th tran and not sold yet)
        # 4. It is meaningless to buy/sell or sell/buy in same day, cuz it doesn't change 
        # anything
        N = len(prices)
        
        if not prices or not K:
            return 0
        
        # If K > N / 2, means you are free to buy/sell (note buy/sell not in same day),
        # so you just needs to buy at every valley and sell at every peak, the max occur
        # of valley/peak is like 1, 3, 2, 5, 1, 6, 3, 7 ...., it requires N / 2 buys and 
        # N / 2 sells. There are peaks require more days to reach, so should wait for that
        # peak to sell. No matther what is the valley/peek case, limit less trans allows
        # to collect every incrase btw each day, so the solution could just simply add up
        # the difference of each day;
        # If N is odd, the result equals N - 1, cuz it will just increase one
        # more buy without sell which should be avoid.
        if K > N // 2:
            ans = 0
            for i, j in zip(prices[1:], prices[:-1]):
                ans += max(0, i - j)
            return ans
        
        dp = [[[-math.inf] * 2 for _ in range(N)] for _ in range(K + 1)]
        dp[0][0][0] = 0
        dp[1][0][1] = -prices[0]
        
        # k starts from 0, because dp[k][i][0] needs to be set to 0, and
        # dp[k][i][1] still -inf. We can do a seperate loop to set that,
        # but to simplify it, we do it toegher with k > 0, but we need
        # to check k > 0 when we set dp[k][i][1]
        ans = 0
        # Trick: It is ok to switch k and n
        for k in range(K + 1):
            for n in range(1, N):
                dp[k][n][0] = max(dp[k][n - 1][0], dp[k][n - 1][1] + prices[n])
                if k > 0:
                    dp[k][n][1] = max(dp[k][n - 1][1], dp[k - 1][n - 1][0] - prices[n])
                ans = max(ans, dp[k][n][0])
        return ans
```

### [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Intuition:
        # mp_1_1: Max profit at tran1 when holding the stock
        # mp_1_0: Max profit at tran1 after sold the stock
        # mp_2_1: Max profit at tran2 when holding/buyting the stock
        # mp_2_0: Max profit at tran2 after sold the stock
        # Initial state at day one:
        # mp_1_1 = -prices[0], cuz we buy 1 stock
        # mp_1_0 = 0, cuz we buy 1 stock and sell it at same day
        # mp_2_1 = -prices[0], cuz we buy 1, sell 1 and buy 1 again
        # mp_2_0 = 0, we repeat buy/sell twice
        mp_1_1, mp_1_0, mp_2_1, mp_2_0 = -prices[0], 0, -prices[0], 0
        
        # We just need prev day's data
        # Starting from day two.
        # State transform:
        # _mp_*_*: cur day; mp_*_*: last day
        # _mp_1_1: 
        #  - do nothing: mp_1_1 - yesterday already hold 1 stock
        #  - buy stock: -p (cuz this is first tran and no stock is held yet)
        # _mp_1_0:
        #  - do nothing: mp_1_0 - yesterday when tran1 already completed
        #  - sell stock: mp_1_1 + p - yesterday when tran1 bought 1 stock
        # _mp_2_1:
        #  - do nothing: mp_2_1
        #  - buy stock: mp_1_0 - p - yesterday already sold tran1 stock
        # _mp_2_0:
        #  - do nothing: mp_2_0
        #  - sell stock: mp_2_1 + p - yesterday alrady bought tran2 stock, today sold it
        for p in prices[1:]:
            _mp_1_1, _mp_1_0, _mp_2_1, _mp_2_0 = max(mp_1_1, -p), max(mp_1_0, mp_1_1 + p), \
                                                 max(mp_2_1, mp_1_0 - p), max(mp_2_0, mp_2_1 + p)

            mp_1_0, mp_1_1, mp_2_0, mp_2_1 = _mp_1_0, _mp_1_1, _mp_2_0, _mp_2_1
        
        return mp_2_0
```

### [309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        N = len(prices)
        dp = [[-math.inf] * 2 for _ in range(N + 1)]
        
        dp[0][0], dp[1] = 0, [0, -prices[0]]
        
        for i in range(2, N + 1):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
            dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i - 1])
        
        return dp[-1][0]

    def maxProfit(self, prices: List[int]) -> int:
        # Intuition: dp[i][j] is the max profit at the END of day i, if 
        # there is j stock in account, j = 0 or 1
        # Use days (prices) as the supporting point, try to calculate dp for each
        # day on each possibilities: sold (has 0 stock) or hold (has 1 stock).
        # When all possibilities on last pre days is ready, we can calc current day.
        # For j = 0 at the end of day i, it might did nothing or sold the stock
        # held before;
        # For j = 1 at the end of day i, it might did nothing or buy the stock.
        N = len(prices)
        # Initial state: before day 1, max profit is 0 if no stock, and it is impossible
        # to hold 1 stock before day 1 (-inf). On day 1, max profit is 0 is no stock,
        # and -prices[0] if hold 1, cuz you have to buy it.
        dp = [[0, -math.inf], [0, -prices[0]]]
        
        for i in range(2, N + 1):
            pre, pre_pre = dp.pop(), dp.pop()
            sold = max(pre[0], pre[1] + prices[i - 1])
            # When calculate the case you have 1 stock be end of day i, 
            # in case you buy it in day i, you have to consider the max
            # profit of not holding stock at DAY i - 2! Cuz you have to
            # wait for a day for cooldown!
            hold = max(pre[1], pre_pre[0] - prices[i - 1])
            dp = [pre, [sold, hold]]
        return dp[-1][0]
```

### [714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        N = len(prices)
        # Initial state at day 1
        hold_0, hold_1 = 0, -prices[0]
        for p in prices[1:]:
            # Fee is only paid when sell the stock
            hold_0 = max(hold_0, hold_1 + p - fee)
            hold_1 = max(hold_1, hold_0 - p)
        return hold_0
```

### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l = len(prices)
        
        valley = float('inf')
        peak = 0
        maxp = 0
        
        # Trick: Find the valley and peak
        for i in range(l):
            p = prices[i]
            if p < valley:
                valley = p
                # When new valley is fund,
                # reset the peak, so the value
                # is not picked by max util
                # new peak is found
                peak = 0
                continue
            elif p > peak:
                peak = p
            maxp = max(maxp, peak - valley)
            
        return maxp
```

### [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i, j in zip(prices[1:], prices[:-1]):
            ans += max(0, i - j)
        return ans
```

### [198. House Robber](https://leetcode.com/problems/house-robber/)

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp = [0] * (len(nums) + 1)
        dp[1] = nums[0]
        
        for i in range(2, len(nums) + 1):
            dp[i] = max(dp[i - 2] + nums[i - 1], dp[i - 1])
            
        return dp[-1]
```

### [213. House Robber II](https://leetcode.com/problems/house-robber-ii/)

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        if len(nums) == 1: return nums[0]
        def max_value(houses):
            dp0, dp1 = 0, houses[0]
            for h in houses[1:]:
                dp0, dp1 = dp1, max(dp1, dp0 + h)
            return dp1
        return max(max_value(nums[:-1]), max_value(nums[1:]))
```


## Number of Subsequences

### [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp is the LIS for array with len i, which must includes i
        dp = [1] * len(nums)
        
        for i in range(1, len(nums)):
            maxval = 0
            for j in range(0, i):
                if nums[i] > nums[j]:
                    maxval = max(maxval, dp[j])
            dp[i] = maxval + 1
            
        maxval = 1
        for i in range(0, len(nums)):
            maxval = max(maxval, dp[i])
            
        return maxval
```

### [403. Frog Jump](https://leetcode.com/problems/frog-jump/)

```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if stones[1] != 1: return False
        N = len(stones)
        dp = {x: set() for x in stones}
        dp[1].add(1)
        
        for x in stones[1:-1]:
            for j in dp[x]:
                for k in range(j - 1, j + 2):
                    if k > 0 and x + k in dp:
                        dp[x + k].add(k)
        
        return bool(dp[stones[-1]])
        
    def canCross(self, stones: List[int]) -> bool:
        N = len(stones)
        if N == 0 or N > 1 and stones[1] != 1:
            return False
        
        dp = [False] * N
        dp[0] = dp[1] = True
        jump_steps = defaultdict(list)
        jump_steps[1] = [1, 2]
        
        for i in range(2, N):
            for j in range(1, i):
                k = stones[i] - stones[j]
                # This make the time complexity O(N^3)
                if k in jump_steps[j]:
                    dp[i] = True
                    jump_steps[i].extend([k - 1, k, k + 1])
        
        return dp[-1]
```

## Longest Increasing Subsequence

### [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # Pattern: LIS - DP
        # dp is the LIS for array with len i, which must includes i
        N = len(nums)
        dp = [1] * N
        for i in range(1, N):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def lengthOfLIS(self, nums):
        # Pattern: LIS - Greedy
        # Intuition: Maintain a monotonous increasing array.
        #   Get x from nums one by one, and add x into dp,
        #   if x is the biggest of dp, append it to end;
        #   if x is not the biggest, replace the FIRST num
        #   in dp that >= x with x; in this way, we are continously
        #   building an mono increasing array with smaller numbers. 
        #   If the new mono array is not as long as cur one, just
        #   replace first no-smaller num, when the numer is enough
        #   it will replace the last largest number and even increase
        #   the complete array. The final array is not a valid mono
        #   array, because some order might not correct, but the length
        #   is the answer.
        dp = []
        for x in nums:
            pos, dp_len = 0, len(dp)
            while pos <= dp_len:
                if pos == dp_len:
                    dp.append(x)
                    break
                elif dp[pos] >= x:
                    dp[pos] = x
                    break
                pos += 1
        return len(dp)
    
    def lengthOfLIS(self, nums):
        dp = []
        def binary_search(x):
            # Trick: hi is len(dp) not len(dp) - 1
            lo, hi = 0, len(dp)
            while lo < hi:
                mid = (hi + lo) // 2
                if dp[mid] < x:
                    lo = mid + 1
                elif dp[mid] >= x:
                    hi = mid
            return lo
        for x in nums:
            i = binary_search(x)
            if i == len(dp):
                dp.append(x)
            else:
                dp[i] = x
        return len(dp)
    
    def lengthOfLIS(self, nums):
        dp = []
        for x in nums:
            i = bisect.bisect_left(dp, x)
            if i == len(dp):
                dp.append(x)
            else:
                dp[i] = x
        return len(dp)
    
    def lengthOfLIS(self, nums):
        # Pattern: LIS Monotonous Array
        #   Use bitsect to mantain a monotonic increasing array
        #   for the cur num, each cur num will replace the smallest
        #   number non-smaller than it self, and it will also keep
        #   record of the max mono array before cur if cur is not
        #   the max.
        dp = [math.inf] * (len(nums) + 1)
        for x in nums:
            dp[bisect.bisect_left(dp, x)] = x
        return dp.index(math.inf)
```


### [354. Russian Doll Envelopes](https://leetcode.com/problems/russian-doll-envelopes/)

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        N = len(envelopes)
        dp = [1] * N
        envelopes.sort()
        for i in range(1, N):
            for j in range(i):
                if envelopes[j][0] < envelopes[i][0] and envelopes[j][1] < envelopes[i][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # Pattern: LIS
        # Trick: Sort by first asc, second desc, so same length env has decending
        #   width so cannot be collected in one LIS, any LIS has ascending width 
        #   and length
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        def lis(nums):
            dp = []
            for x in nums:
                i = bisect.bisect_left(dp, x)
                if i == len(dp):
                    dp.append(x)
                else:
                    dp[i] = x
            return len(dp)
        return lis([x[1] for x in envelopes])
```

### [1671. Minimum Number of Removals to Make Mountain Array](https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/)

```python
class Solution:
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        N = len(nums)
        dp, dp2 = [1] * N, [1] * N
        for i in range(1, N):
            for j in range(i):
                if nums[i] > nums[j]:
                    # If num i > num j, update the LIS length, no impact on
                    # mountain length
                    dp[i] = max(dp[i], dp[j] + 1)
                elif nums[i] < nums[j]:
                    if dp[j] > 1:
                        # First check the case of LIS till j, if there is LIS 
                        # (dp1[j] > 1) till j, i becomes the first descending 
                        # number.
                        dp2[i] = max(dp2[i], dp[j] + 1)
                    if dp2[j] > 1:
                        # Second check the case of LDS till j, if there is LDS
                        # (dp2[j] > 1) till j, i will be adding to the descending
                        # sequence.
                        dp2[i] = max(dp2[i], dp2[j] + 1)
        
        return N - max(dp2)
    
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        def lis(arr):
            dp = [math.inf] * (len(arr) + 1)
            for x in arr:
                dp[bisect.bisect_left(dp, x)] = x
            return dp.index(math.inf)
        
        N, ans = len(nums), 0
        # Make sure i is the biggeist number in both left and right subarray
        # this will make sure the max LIS includes i; reverse right array and
        # get max LIS for both left and right and reverse right result;
        # we can make sure i is included in both and we just have to compare
        # each i with length a + b -1 (removing duplicate i)
        for i in range(1, N - 1):
            left  = [n for n in nums[:i] if n < nums[i]] + [nums[i]]
            right = [nums[i]] + [n for n in nums[i + 1:] if n < nums[i]]
            right = right[::-1]
            a, b = lis(left), lis(right)
            if a > 1 and b > 1:
                ans = max(ans, a + b - 1)
        
        return N - ans
    
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        def lis(arr):
            # Instead of return the max len, lens[i] is the max len of
            # LIS end with i AND include i.
            N = len(arr)
            dp = [math.inf] * (N + 1)
            lens = [1] * N
            for i, x in enumerate(arr):
                pos = bisect.bisect_left(dp, x)
                lens[i] = pos + 1
                dp[pos] = x
            return lens
        lis_a, lis_b = lis(nums), lis(nums[::-1])[::-1]
        ans, N = 0, len(nums)
        for i in range(N):
            if lis_a[i] > 1 and lis_b[i] > 1:
                ans = max(ans, lis_a[i] + lis_b[i] - 1)
        
        return N - ans
```

### [403. Frog Jump](https://leetcode.com/problems/frog-jump/)

```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if stones[1] != 1: return False
        N = len(stones)
        dp = {x: set() for x in stones}
        dp[1].add(1)
        
        for x in stones[1:-1]:
            for j in dp[x]:
                for k in range(j - 1, j + 2):
                    if k > 0 and x + k in dp:
                        dp[x + k].add(k)
        
        return bool(dp[stones[-1]])
        
    def canCross(self, stones: List[int]) -> bool:
        N = len(stones)
        if N == 0 or N > 1 and stones[1] != 1:
            return False
        
        dp = [False] * N
        dp[0] = dp[1] = True
        jump_steps = defaultdict(list)
        jump_steps[1] = [1, 2]
        
        for i in range(2, N):
            for j in range(1, i):
                k = stones[i] - stones[j]
                # This make the time complexity O(N^3)
                if k in jump_steps[j]:
                    dp[i] = True
                    jump_steps[i].extend([k - 1, k, k + 1])
        
        return dp[-1]
```


### [646. Maximum Length of Pair Chain](https://leetcode.com/problems/maximum-length-of-pair-chain/submissions/)
```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        N = len(pairs)
        dp = [1] * N
        pairs.sort()
        for i in range(1, N):
            for j in range(i):
                if pairs[j][1] < pairs[i][0]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return dp[-1]
    
    def findLongestChain(self, pairs):
        cur, ans = float('-inf'), 0
        # Trick: operator.itemgetter(1)
        for x, y in sorted(pairs, key = operator.itemgetter(1)):
            if cur < x:
                cur = y
                ans += 1
        return ans
```

### [673. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)
```python
class Solution:                   
    def findNumberOfLIS(self, nums: List[int]) -> int:
        N = len(nums)
        if N <= 1: return N
        # Intuition: 
        #   lengths[i] -  the max len of LIS ending with i
        #   counts[i]  -  the count of LIS ending with i with the max len
        #   Starting from 1 to N - 1, for each i, iterate j in [0, i - 1] looking
        #   for nums[i] > nums[j]. For any matching j, we can build lengths[i] and 
        #   counts[i].
        #   For one j, if lengths[j] >= lengths[i], this means previous values for
        #   lengths[i] and counts[i] is not useful, we found completely different
        #   set of LIS which are longer, so we just replace lengths[i] and counts[i];
        #   if lengths[j] + 1 = lengths[i], means we find more LIS for i with same
        #   len as lengths[i], so just add to current counts[i], but lengths[i] remain
        #   unchanged. We are iterate [0, i - 1], so no worry on duplicates.
        lengths, counts = [1] * N, [1] * N
        for i in range(1, N):
            for j in range(i):
                if nums[i] > nums[j]:
                    if lengths[j] >= lengths[i]:
                        lengths[i] = lengths[j] + 1
                        counts[i] = counts[j]
                    elif lengths[j] + 1 == lengths[i]:
                        counts[i] += counts[j]
        longest = max(lengths)
        return sum(c for i, c in enumerate(counts) if lengths[i] == longest)
```
### [845. Longest Mountain in Array](https://leetcode.com/problems/longest-mountain-in-array/)

```python
class Solution:
    # 53 Maximum Subarray
    # 121 Best Time to Buy and Sell Stock
    # 152 Maximum Product Subarray
    # 238 Product of Array Except Self
    # 739 Daily Temperatures
    # 769 Max Chunks to Make Sorted
    # 770 Max Chunks to Make Sorted II
    # 821 Shortest Distance to a Character
    # 845 Longest Mountain in Array
    def longestMountain(self, arr: List[int]) -> int:
        N = len(arr)
        up, down = [0] * N, [0] * N
        for i in range(1, N):
            if arr[i - 1] < arr[i]:
                up[i] = up[i - 1] + 1
        for i in range(N - 2, -1, -1):
            if arr[i + 1] < arr[i]:
                down[i] = down[i + 1] + 1
        return max([u + d + 1 for u, d in zip(up, down) if u and d] or [0])
    
    def longestMountain1(self, arr: List[int]) -> int:
        # One pass
        pos, N, ans = 1, len(arr), 0
        while pos < N:
            while pos < N and arr[pos] == arr[pos - 1]:
                pos += 1
            
            up = 0
            while pos < N and arr[pos] > arr[pos - 1]:
                up += 1
                pos += 1
            
            down = 0
            while pos < N and arr[pos] < arr[pos - 1]:
                down += 1
                pos += 1
                
            if up and down:
                ans = max(ans, up + down + 1)
        return ans
```


## Jump Game

### [55. Jump Game](https://leetcode.com/problems/jump-game/)

```python
class Solution:
    
    def canJump(self, nums: List[int]) -> bool:
        N = len(nums)
        dp = [True] + [False] * (N - 1)
        for i, x in enumerate(nums):
            for j in range(min(N - 1, i + 1), min(N - 1, i + x) + 1):
                dp[j] = dp[i]
        return dp[-1]
    
    def canJump(self, nums: List[int]) -> bool:
        N = len(nums)
        dp = [True] + [False] * (N - 1)
        good_pos = 0
        for i, x in enumerate(nums):
            if dp[i]:
                for j in range(good_pos, min(N - 1, i + x) + 1):
                    dp[j] = True
                good_pos = min(N - 1, i + x)
        return dp[-1]
    
    def canJump(self, nums: List[int]) -> bool:
        # Alg: Greedy
        # Tuition: The final target is last index, if a pos can go
        #   to target, the pos is good. If a pos can reach to a good
        #   pos, it is good too; if a pos can reach to target and 
        #   it is on left of another pos, it can reach that pos too;
        #   if a pos can reach to a good pos, it can reach another
        #   good pos in middle. If a pos cannot reach it's leftmost (nearest)
        #   good pos, it cannot reach to other good pos and target neither.
        #   So starting from right most index, we find the leftmost good pos,
        #   then iteratively find next good pos which can reach to cur good pos.
        #   the last good pos should be 0.
        
        N = len(nums)
        good_pos = N - 1
        for i in range(N - 2, -1, -1):
            if nums[i] >= good_pos - i:
                good_pos = i
        return good_pos == 0
```

### [1326. Minimum Number of Taps to Open to Water a Garden](https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/)

```python
class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        # Intuition: dp[i] is min tap required to water [1, i] gardon.
        #   total garden is [1, n], total tap is [0, n], there are n
        #   gardens and n + 1 taps.
        #   Go through each tap[i]'s scope, for every dp[j] in scope, 
        #   the ans is dp[k] + 1 where k is last garden out of left
        #   border of scope. For each i, updating min of dp[j], you find
        #   the final min for dp[j].
        # Pattern: DP - A lot of cases iterate items as support point
        dp = [0] + [n + 2] * n
        for i, x in enumerate(ranges):
            for j in range(max(0, i - x + 1), min(n, i + x) + 1):
                dp[j] = min(dp[j], dp[max(0, i - x)] + 1)
        
        return dp[-1] if dp[-1] < n + 2 else -1
```

### [1340. Jump Game V](https://leetcode.com/problems/jump-game-v/)

```python
class Solution:
    def maxJumps(self, arr: List[int], d: int) -> int:
        # DP - Bottom Up O(nLogn)
        N = len(arr)
        # dp[i] is the max num of indices start from i
        dp = [1] * N
        # Trick: Sort by height, lowest has ans as 1. Use height
        # as supporting point
        sorted_arr = sorted([(x, i) for i, x in enumerate(arr)])
        for x, i in sorted_arr[1:]:
            for j in range(min(N -1, i + 1), min(N - 1, i + d) + 1):
                if arr[j] >= x:
                    break
                dp[i] = max(dp[i], dp[j] + 1)

            for j in range(max(0, i - 1), max(0, i - d) - 1, -1):
                if arr[j] >= x:
                    break
                dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def maxJumps(self, arr: List[int], d: int) -> int:
        # DP - Top Down O(n)
        N = len(arr)
        @functools.lru_cache(None)
        def dp(i):
            ans = 1
            for j in range(min(N -1, i + 1), min(N - 1, i + d) + 1):
                if arr[j] >= arr[i]:
                    break
                ans = max(ans, dp(j) + 1)
            for j in range(max(0, i - 1), max(0, i - d) - 1, -1):
                if arr[j] >= arr[i]:
                    break
                ans = max(ans, dp(j) + 1)
            return ans
        return max(map(dp, range(N)))
```

### [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        N = len(nums)
        dp = [0] + [N] * (N - 1)
        for i, x in enumerate(nums):
            for j in range(min(N - 1, i + 1), min(N - 1, i + x) + 1):
                dp[j] = min(dp[j], dp[i] + 1)
        return dp[-1]
    
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1: return 0
        left, right, steps = 1, nums[0], 1
        while right < len(nums) - 1:
            steps += 1
            left, right = right + 1, max([i + nums[i] for i in range(left,right + 1)])
        return steps
```

### [1696. Jump Game VI](https://leetcode.com/problems/jump-game-vi/)

```python
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        # dp is the max score can achieve at step i
        N = len(nums)
        dp = [0] * N
        dp[0] = nums[0]
        
        # q is monotonicc decreasing queue for prev best scores for step
        # [i - k, i - 1]
        q = deque([0])
        for i in range(1, N):
            # For each step, get rid of pre step cannot be reached back
            if q[0] < i - k:
                q.popleft()
            
            # The head of monotonic decreasing stack/queue is max score
            # dp[i] = max(dp[i - k],...,dp[i - 1]) + nums[i]
            # max is q[0]
            dp[i] = dp[q[0]] + nums[i]
            
            # Update the mono queue using current dp[i]
            # > and >= both work
            while q and dp[i] > dp[q[-1]]:
                q.pop()
            q.append(i)
        
        return dp[-1]
```

