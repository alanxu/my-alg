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

### [805. Split Array With Same Average](https://leetcode.com/problems/split-array-with-same-average/)

```python
class Solution:
    def splitArraySameAverage(self, nums: List[int]) -> bool:
        # TLE
        # https://youtu.be/tMaWUhj5YaU
        N, total = len(nums), sum(nums)
        
        # Intuition: dp[sum][num] is whether we can find a subset of num items
        # in nums that sums to sum.
        # For each num calc the whole table for that num agaist all nums. Why it
        # works? Becuz we just need true or false, say (a, c, e)(b, d, f), we know
        # (a, c, e) can make it, if we run a and c cannot get the result, but e
        # can, if we run a and e first then c can. So the last one of the subset
        # will lead to a True if matches.
        
        # If average(A) = average(B), average(A) = average(B) = average(All)
        # total / N = sum / num
        dp = [[False] * (N + 1) for _ in range(total + 1)]
        dp[0][0] = True
        
        for x in nums:
            # If we run a loop outside of the dp loop, there is a typical situation
            # that the dp needs to referecing to the previous state that already
            # changed by current round. 2 solutions:
            # - use 2 dp and copy it
            # - iterate dp reversed order (next solution)
            dp2 = copy.deepcopy(dp)
            for ssum in range(x, total + 1):
                for num in range(1, N):
                    # If Use current x can find a sum and num subset,
                    # check if this sum and num can be the equal average
                    if dp2[ssum - x][num - 1]:
                        dp[ssum][num] = True
                        if ssum * N == total * num:
                            return True
        return False
    
    def splitArraySameAverage(self, nums: List[int]) -> bool:
        # TLE
        N, total = len(nums), sum(nums)
        nums.sort()
        dp = [[False] * (N + 1) for _ in range(total + 1)]
        dp[0][0] = True
        
        cur_sum = 0
        for x in nums:
            cur_sum += x
            # For cur x, the upper capacity is the running sum so far,
            # more than that should not be considered.
            # For num, limit it to N//2 + 1, + 1 to be safe, cuz we
            # just want to find the smaller half.
            # Trick: Use reversed order to get ride of 2 dp table
            for ssum in range(cur_sum, x - 1, -1):
                for num in range(N//2 + 1, 0, -1):
                    if dp[ssum - x][num - 1]:
                        dp[ssum][num] = True
                        if num != N and ssum * N == total * num:
                            return True
        return False

    def splitArraySameAverage(self, A: List[int]) -> bool:
        # Dont understand
        # A subfunction that see if total k elements sums to target
        # target is the goal, k is the number of elements in set B, i is the index we 
        # have traversed through so far
        mem = {}

        def find(target, k, i):
            # if we are down searching for k elements in the array, 
            # see if the target is 0 or not. This is a basecase
            if k == 0: return target == 0

            # if the to-be selected elements in B (k) + elements we have traversed so 
            # far is larger than total length of A
            # even if we choose all elements, we don't have enough elements left, 
            # there should be no valid answer.
            if k + i > len(A): return False

            if (target, k, i) in mem: return mem[(target, k, i)]

            # if we choose the ith element, the target becomes target - A[i] for total sum
            # if we don't choose the ith element, the target doesn't change
            mem[(target - A[i], k - 1, i + 1)] = find(target - A[i], k - 1, i + 1) or find(target, k, i + 1)

            return mem[(target - A[i], k - 1, i + 1)]

        n, s = len(A), sum(A)
        # Note that the smaller set has length j ranging from 1 to n//2+1
        # we iterate for each possible length j of array B from length 1 to length n//2+1
        # if s*j%n, which is the sum of the subset, it should be an integer, so we only 
        # proceed to check if s * j % n == 0
        # we check if we can find target sum s*j//n (total sum of j elements that sums to s*j//n)
        return any(find(s * j // n, j, 0) for j in range(1, n // 2 + 1) if s * j % n == 0)
```
