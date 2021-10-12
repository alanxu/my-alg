
#

DP about subarray or subseq
DP about make choices based on some limitations

dp[i] is the answer of subarray [0,i] ending with i

Finally aggregate on all dp values


## Kadane's Algorithm - Max Subarray Sum

### [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

```python
class Solution:
    
    def maxSubArray(self, nums: List[int]) -> int:
        N = len(nums)
        # dp[i] is max subarray sum ending with i
        dp = [0] * N
        dp[0] = nums[0]
        for i in range(1, N):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)
    
    def maxSubArray(self, nums: List[int]) -> int:
        # Alg: Kadane's Algorithm
        #   It is just state compression of regular DP
        global_max = local_max = nums[0]
        for x in nums[1:]:
            local_max = max(local_max + x, x)
            global_max = max (global_max, local_max)
        return global_max
```

### [1186. Maximum Subarray Sum with One Deletion](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/)

```python
class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        # Pattern: A lot of DP questions about subseq or subarray, can be resolved
        #   by defining the state by dp[i] ending at i, then 
        N = len(arr)
        # Initial value for deleted[0] is not 0, it will has issue for all negative case
        deleted, no_delete = [-math.inf] * N, [arr[0]] + [-math.inf] * (N + 1)
        
        for i in range(1, N):
            deleted[i] = max(no_delete[i - 1], deleted[i - 1] + arr[i])
            # No delete can be arr[i] itself or plus the pre max
            no_delete[i] = max(no_delete[i - 1] + arr[i], arr[i])
        
        return max(deleted + no_delete)
```

### [1749. Maximum Absolute Sum of Any Subarray](https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/)

```python
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        # Alg: Kadane's Algo to get max greater than 0 and min
        # less than 0
        # Different from traditional Kadane's algo, the initial
        # value is 0 and loop starting from i == 0
        #
        # Another way to think about it is whenever the sum 
        # more/less than 0 by adding x, dump x and start over
        ans = mx = mn = 0
        for x in nums:
            mx = max(0, mx + x)
            mn = min(0, mn + x)
            ans = max(ans, mx, -mn)
        return ans
    
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        # Alg: Prefix Sum
        # The max abs subarray sum is the difference btw
        # max/min prefix sum. Usually a subarray sum is
        # difference of a prefix sum and another one before
        # it, in this case, we don't care becuz it is abs
        mx = mn = ps = 0
        for x in nums:
            ps += x
            mx, mn = max(mx, ps), min(mn, ps)
        return mx - mn
    
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        # 1 line version of Prefix Sum solution
        return max(accumulate(nums, initial=0)) - min(accumulate(nums, initial=0))
```

### [152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

```python
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_v, min_v, res = nums[0], nums[0], nums[0]
        
        for i in range(1, len(nums)):
            num = nums[i]
            max_v_cur = max_v * num
            min_v_cur = min_v * num
            max_v = max(max_v_cur, min_v_cur, num)
            min_v = min(max_v_cur, min_v_cur, num)
            
            res = max(res, max_v)
        
        return res
```

### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        local_max, global_max = 1, 1
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                if nums[i] == nums[i - 1] + 1:
                    local_max += 1
                    global_max = max(global_max, local_max)
                else:
                    local_max = 1
        return global_max

    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        nums.sort()
        N = len(nums)
        dp = [1] * N
        for i in range(1, N):
            if nums[i] == nums[i - 1]:
                dp[i] = dp[i - 1]
            elif nums[i] == nums[i - 1] + 1:
                dp[i] = dp[i - 1] + 1

        return max(dp)
```

### [674. Longest Continuous Increasing Subsequence](https://leetcode.com/problems/longest-continuous-increasing-subsequence/)

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        local_max = global_max = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                local_max += 1
                global_max = max(global_max, local_max)
            else:
                local_max = 1
        return global_max
```

### [1191. K-Concatenation Maximum Sum](https://leetcode.com/problems/k-concatenation-maximum-sum/)


## Stairs

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


## Others

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

### [376. Wiggle Subsequence](https://leetcode.com/problems/wiggle-subsequence/)

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        N = len(nums)
        # Intuition: https://youtu.be/j7U3olaBxMg
        # low[i] - The max len of wiggle subseq for index [0,i] where the last
        #          num is a valley, the last index i might or might not be the last 
        #          in subseq. 
        #          * If nums[i] <=  nums[i - 1], it just deepen the valey
        #          but the num of peek/valley not change, but we can use the num[i]
        #          as the new valley, we MUST do that so the alg can work. 
        #          * If nums[i] > nums[i - 1], nums[i] forms the last peak, so low[i]
        #          is ending with i - 1 not i
        # high[i] - The max len of wiggle subseq for index [0,i] where the last
        #          num is a peek, the last index i might or might not be the last 
        #          in subseq. 
        low, high = [1] * N, [1] * N
        
        for i in range(1, N):
            if nums[i] > nums[i - 1]:
                high[i] = low[i - 1] + 1
                low[i] = low[i - 1]
            elif nums[i] < nums[i - 1]:
                low[i] = high[i - 1] + 1
                high[i] = high[i - 1]
            else:
                high[i] = high[i - 1]
                low[i] = low[i - 1]
        
        return max(low[-1], high[-1])
```

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

### [1289. Minimum Falling Path Sum II](https://leetcode.com/problems/minimum-falling-path-sum-ii/)

```python
class Solution:
    def minFallingPathSum(self, arr: List[List[int]]) -> int:
        # Intuition: Update arr in place, for each cell, calculate
        #   min accumulate sum until cur cell.
        R, C = len(arr), len(arr[0])
        
        for r in range(1, R):
            for c in range(C):
                # Min accumulate sum to cur cell is cur val plus min
                # of all cells in prevous row excluding the cell above.
                # This can be optimized to avoid repeatedly calc min.
                arr[r][c] += min(arr[r - 1][:c] + arr[r - 1][c + 1:])
                
        return min(arr[-1])
    
    def minFallingPathSum(self, arr: List[List[int]]) -> int:
        R, C = len(arr), len(arr[0])
        dp = [float('inf') for _ in range(C)]
        
        def find_two_smallest(a):
            """
            returns: 
                min1: the smallest value in the row
                min2: the 2nd smallest value in the row
                i1:   the index if the smallest in the row
            """
            min1, min2 = float('inf'), float('inf')
            for i, x in enumerate(a):
                if x <= min1:
                    min1, min2, i1 = x, min1, i
                elif x < min2:
                    min2 = x
            return [min1, min2, i1]
        
        d = find_two_smallest(arr[0])
        for r in range(1, R):
            for c in range(C):
                if d[2] == c:
                    dp[c] = d[1] + arr[r][c]
                else:
                    dp[c] = d[0] + arr[r][c]
            d = find_two_smallest(dp)
        return d[0]
    
    def minFallingPathSum(self, arr: List[List[int]]) -> int:
        R, C = len(arr), len(arr[0])
        
        for r in range(1, R):
            # Trick: sort prev row before calc cur row to avoid duplicated calc
            accu_sums = sorted(arr[r - 1])
            min_idx = arr[r - 1].index(accu_sums[0])
            for c in range(C):
                # If the above cell of cur cell is min, choose 2nd min
                arr[r][c] += accu_sums[1] if c == min_idx else accu_sums[0]

        return min(arr[-1])
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

### [487. Max Consecutive Ones II](https://leetcode.com/problems/max-consecutive-ones-ii/)

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        # Intuition: DP
        #   Type1, only related i and i - 1.
        N = len(nums)
        # flip[i]:    For [0,i], max len of 1's with 1 flip
        # no_flip[i]: For [0, i], max len of 1's without flip
        flip, no_flip = [0] * (N + 1), [0] * (N + 1)
        for i in range(1, N + 1):
            if nums[i - 1]:
                # No need to flip i, so flip[i - 1] + 1 and no_flip[i - 1]+ 1
                flip[i] = flip[i - 1] + 1
                no_flip[i] = no_flip[i - 1] + 1
            else:
                # If nums[i] is 0, it needs to be fliped to achieve max len flip[i]
                flip[i] = no_flip[i - 1] + 1
                
                # If nums[i] is 0, without flip, the seq will end with 0, so len of 1's is 0
                no_flip[i] = 0

        return max(flip + no_flip)
```

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
        # dp[k][i][1] still -inf. Here we use same loops to set value for
        # all scenarios when k=0, rather than set it as initial value.
        # We can do a seperate loop to set that,
        # but to simplify it, we do it toegher with k > 0, but we need
        # to check k > 0 when we set dp[k][i][1].
        # n starts from 1, because initial values for day 0 already set
        # for k > 1, all -math.inf for day 0.
        # We check k > 0, before calc dp[k][n][1], because at least 1
        # transaction is required to hold 1 stock.
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

### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        l = len(nums)
        dp_l, dp_r, ans = [0] * l, [0] * l, [0] * l
        dp_l[0] = 1
        dp_r[l - 1] = 1
        
        for i in range(1, l):
            dp_l[i] = dp_l[i - 1] * nums[i - 1]
            
        for i in reversed(range(l - 1)):
            dp_r[i] = dp_r[i + 1] * nums[i + 1]
        
        for i in range(l):
            ans[i] = dp_l[i] * dp_r[i]
        
        return ans
```

### [639. Decode Ways II](https://leetcode.com/problems/decode-ways-ii/)

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if s == '0': return 0
        s = '00' + s
        N, MOD = len(s), 10 ** 9 + 7
        dp = [1] * N
        for i in range(2, N):
            if s[i] == '0':
                if s[i - 1] == '*':
                    dp[i] = dp[i - 2] * 2
                elif s[i - 1] in ('1', '2'):
                    dp[i] = dp[i - 2]
                else:
                    return 0
            elif s[i] == '*':
                dp[i] = dp[i - 1] * 9
                if s[i - 1] == '1':
                    dp[i] += dp[i - 2] * 9
                elif s[i - 1] == '2':
                    dp[i] += dp[i - 2] * 6
                elif s[i - 1] == '*':
                    dp[i] += dp[i - 2] * 15
            else:
                dp[i] = dp[i - 1]
                    
                if s[i - 1] == '1':
                    dp[i] += dp[i - 2]
                elif s[i - 1] == '2' and s[i] <= '6':
                    dp[i] += dp[i - 2]
                elif s[i - 1] == '*':
                    if '1' <= s[i] <= '6':
                        dp[i] += dp[i - 2] * 2
                    else:
                        dp[i] += dp[i - 2]
                
            dp[i] %= MOD

        return dp[-1]
```

### [600. Non-negative Integers without Consecutive Ones](https://leetcode.com/problems/non-negative-integers-without-consecutive-ones/)

```python
class Solution:
    def findIntegers(self, num: int) -> int:
        # Build reference data
        # dp[i] denotes num of cases for a i-digits binary without adjacent 1's
        # Use 33 to make it 1-indexed.
        dp = [0] * 33
        dp[0], dp[1] = 1, 2
        for i in range(2, 33):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        # Build the threshold number into binary array,
        # the array is in reversed order
        # Trick: Convert a int into binary
        digits = [0] * 33
        for i in range(1, 33):
            digits[i] = num % 2
            num = num // 2
        
        # Iterate threshod binary from higher digit, based
        # on location of 1's, we calc cases
        i, ans = 32, 0
        while(i >= 1):
            
            if digits[i] == 0:
                # If the benchmark digits is 0 at i, our ans has to be 0 at i too
                # so continue search for 1 in benchmark so we can calc cases
                i -= 1
            else:
                # If the benchmark digits is 1 at i, there are two type in our ans
                # 1. 0 at i, all digits after that is free, so totacl case of this
                # type is dp[i - 1];
                # 2. 1 at i, same as benchmark, so we have to continue searching next
                # 1 in benchmark... 
                ans += dp[i - 1]
                
                if i >= 2 and digits[i - 1] == 1:
                    ans += dp[i - 2]
                    return ans
                else:
                    i -= 2
        
        # +1 is for the case that equals to the benchmark
        return ans + 1
```

### [656. Coin Path](https://leetcode.com/problems/coin-path/)
```python
class Solution:
    def cheapestJump(self, A: List[int], B: int) -> List[int]:
        # Pattern: Partition 2
        # Intution: First glance this is Partition 2 not sure why. Then
        # when look at the formular, it only require first part to be subproblem,
        # so Partition 2 is not good, cuz it is O(bn^2)
        # TLE
        A = [-1] + A
        N = len(A)
        dp = [[math.inf] * N for _ in range(N)]
        for i in range(1, N):
            if A[i] != -1:
                dp[i][i] = A[i]
        
        ans = [[math.inf]] * N
        ans[1] = [1]
        
        for l in range(2, N):
            for i in range(1, N - l + 1):
                j = i + l - 1
                if A[i] != -1 and A[j] != -1:
                    k_cache = {}
                    for k in range(max(i, j - B), j):
                        # dp[i][j] = min(dp[i][j], dp[i][k] + A[j])
                        # if dp[i][j] == dp[i][k] + A[j] and dp[i][j] != math.inf and i == 1:
                        #     path = ans[k] + [j]
                        #     if path < ans[j]:
                        #         ans[j] = path
                                
                        if dp[i][k] + A[j] < dp[i][j]:
                            dp[i][j] = dp[i][k] + A[j]
                            if i == 1:
                                ans[j] = ans[k] + [j]
                        elif dp[i][k] + A[j] == dp[i][j] and i == 1:
                            path = ans[k] + [j]
                            # print(path)
                            if path < ans[j]:
                                ans[j] = path

        return ans[-1] if ans[-1] < [math.inf] else []

    def cheapestJump(self, A: List[int], B: int) -> List[int]:
        A = [-1] + A
        N = len(A)
        # Pattern: DP - Type 2
        # Because only first part of formula is subproblem, so not partition 2.
        # Intuition: dp[i] is min cost jump from 1 to i
        dp = [math.inf] * N
        dp[1] = A[1]
        # No need to give default value [math.inf], cus the flow is
        # controled by dp, paths will be updated when it should be;
        # For A[i] == inf, dp[i] == dp[j] + A[j] will match, but
        # paths[i] will not changed, cuz it is already smallest
        paths = [[]] * N
        paths[1] = [1]
        
        for i in range(2, N):
            # Only need to check i not j, cuz
            # j was i before
            if A[i] == -1:
                continue
            for j in range(max(1, i - B), i):
                # Trick: inf == inf + 1 -> true
                if dp[i] > dp[j] + A[j]:
                    # If find a optimal j before i, update dp and paths
                    # this is the first value for the new optimal ans
                    dp[i] = dp[j] + A[j]
                    paths[i] = paths[j] + [i]
                elif dp[i] == dp[j] + A[j]:
                    # If another j has same optimal value, compare the path.
                    # There is another possibility for dp[j] == inf, the code
                    # will be executed, paths will not be updated
                    path = paths[j] + [i]
                    if path < paths[i]:
                        paths[i] = path
        
        return paths[-1]
```

### [920. Number of Music Playlists](https://leetcode.com/problems/number-of-music-playlists/)

```python
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        
        @functools.lru_cache(None)
        def dp(n, l):
            # Parttern: DP - Recursion + Memo - How many ways
            # This is usually type 1 DP
            # Think about two types: Rob/NoRob, Select/NoSelect, Used/NoUsed
            # Think about order of shuaiguo: first + dp(), or dp() + last
            # Anwser is sum of all types
            
            # Intuition: Consider current recursion is after a bunch of previous
            # Two types: 
            # - First time use a song
            # - The song is used before
            # The two type are exclusive so should be added to make an answer
            
            # There are n song, cannot use n song to make a empty list
            if l == 0 and n  > 0: return 0
            # There are empty song, cannot use empty song to make a list of l > 0
            if l  > 0 and n == 0: return 0
            # There are empty song and empty list, perfect 1 match
            if l == 0 and n == 0: return 1
            
            # For the current resursion, shuaiguo prevous selection to dp(), and just
            # select the last song, select a song not used before
            ans = dp(n - 1, l - 1) * (N - (n - 1))
            
            # There is other possiblilities if choose a used song. In this case,
            # we look for dp(n, l - 1), all n song is used before. And the selection
            # of last song cannot be recent k, if there is less than k books in the 
            # current recursion, this case is not valid
            ans += dp(n, l - 1) * max(n - K, 0)
            
            return ans % (10 ** 9 + 7)
        
        return dp(N, L)
```

### [1639. Number of Ways to Form a Target String Given a Dictionary](https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/)

```python
class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        MOD = 10 ** 9 + 7
        N, K = len(target), len(words[0])
        
        # Flaten words to 1D array with counts and 1-indexed
        counts = [Counter() for _ in range(K + 1)] 
        for c in range(K):
            for r in range(len(words)):
                counts[c + 1][words[r][c]] += 1

        # Trick: Make target 1-indexed
        target = '?' + target
        
        # dp[i][k] denotes the ways to form target[1,i] using flaten words[1, k]
        dp = [[0] * (K + 1) for _ in range(N + 1)]
        # Determine initial value later
        for k in range(K + 1):
            dp[0][k] = 1
        
        for i in range(1, N + 1):
            for k in range(1, K + 1):
                # If dont use kth in words
                dp[i][k] = dp[i][k - 1]
                
                # If use kth in words and it can form target[i]
                if target[i] in counts[k]:
                    count = counts[k][target[i]]
                    dp[i][k] += dp[i - 1][k - 1] * count
                dp[i][k] %= MOD
        
        return dp[-1][-1] % MOD
```

### [1692. Count Ways to Distribute Candies](https://leetcode.com/problems/count-ways-to-distribute-candies/)

```python
class Solution:
    def waysToDistribute(self, n: int, k: int) -> int:
        MOD = 10 ** 9 + 7
        
        # dp[i][j] denotes ways to put candiies[1, i] in j bags
        # bags has no difference
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # How to determin the initial value? Draw dp table?
        for i in range(1, n + 1):
            dp[i][1] = 1
        
        for i in range(1, n + 1):
            for j in range(2, k + 1):
                # If put ith candy in a new bag, it is dp[i - 1][j - 1];
                # If put ith candy in existing bag, j bags alredy used,
                # there are j ways to put ith candy, j * dp[i - 1][j]
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j] * j
                dp[i][j] %= MOD
        
        return dp[-1][-1] % MOD
```

### [1406. Stone Game III](https://leetcode.com/problems/stone-game-iii/)

```python
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        A = stoneValue
        N, total = len(A), sum(A)
        dp = [-math.inf] * N + [0]
        dp[N - 1] = A[-1]
        
        for i in range(N - 2, -1, -1):
            dp[i] = max(dp[i], sum(A[i: i + 1]) + sum(A[i + 1:]) - dp[i + 1])
            if i <= N - 2:
                dp[i] = max(dp[i], sum(A[i:i + 2]) + sum(A[i + 2:]) - dp[i + 2])
            if i <= N - 3:
                dp[i] = max(dp[i], sum(A[i:i + 3]) + sum(A[i + 3:]) - dp[i + 3])
        
        if 2 * dp[0] > total: return 'Alice'
        if 2 * dp[0] < total: return 'Bob'
        
        return 'Tie'
    
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        A = stoneValue
        N, total = len(A), sum(A)
        dp = [-math.inf] * N + [0]
        dp[N - 1] = A[-1]
        
        for i in range(N - 2, -1, -1):
            dp[i] = max(dp[i], sum(A[i:]) - dp[i + 1])
            if i <= N - 2:
                dp[i] = max(dp[i], sum(A[i:]) - dp[i + 2])
            if i <= N - 3:
                dp[i] = max(dp[i], sum(A[i:]) - dp[i + 3])
        
        if 2 * dp[0] > total: return 'Alice'
        if 2 * dp[0] < total: return 'Bob'
        
        return 'Tie'
    
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        A = stoneValue
        N, total = len(A), sum(A)
        dp = [-math.inf] * N + [0]
        dp[N - 1] = A[-1]
        
        prefix_sum = A[-1]
        for i in range(N - 2, -1, -1):
            prefix_sum += A[i]
            dp[i] = max(dp[i], prefix_sum - dp[i + 1])
            if i <= N - 2:
                dp[i] = max(dp[i], prefix_sum - dp[i + 2])
            if i <= N - 3:
                dp[i] = max(dp[i], prefix_sum - dp[i + 3])
        
        if 2 * dp[0] > total: return 'Alice'
        if 2 * dp[0] < total: return 'Bob'
        
        return 'Tie'
    
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        A = stoneValue
        N, total = len(A), sum(A)
        dp = [-math.inf] * N + [0]
        dp[N - 1] = A[-1]
        
        prefix_sum = A[-1]
        for i in range(N - 2, -1, -1):
            prefix_sum += A[i]
            for k in range(1, min(3, N - i) + 1):
                dp[i] = max(dp[i], prefix_sum - dp[i + k])
        
        print(dp)
        
        if 2 * dp[0] > total: return 'Alice'
        if 2 * dp[0] < total: return 'Bob'
        
        return 'Tie'
    
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        A = stoneValue
        N, total = len(A), sum(A)
        dp = [-math.inf] * N + [0]
        
        prefix_sum = 0
        # We found N - 1 can be added to the loop, it can give
        # some benefits for further improvement, as all steps
        # are handled in same way
        for i in range(N - 1, -1, -1):
            prefix_sum += A[i]
            for k in range(1, min(3, N - i) + 1):
                dp[i] = max(dp[i], prefix_sum - dp[i + k])
        
        if 2 * dp[0] > total: return 'Alice'
        if 2 * dp[0] < total: return 'Bob'
        
        return 'Tie'
    
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        A = stoneValue
        N, total = len(A), sum(A)
        # Add 3 dummy states
        dp = [-math.inf] * N + [0] * 3
        
        prefix_sum = 0

        for i in range(N - 1, -1, -1):
            prefix_sum += A[i]
            # We can remove the  min(3, N - i) by adding 3 dummy states
            for k in range(1, 3 + 1):
                dp[i] = max(dp[i], prefix_sum - dp[i + k])
        
        if 2 * dp[0] > total: return 'Alice'
        if 2 * dp[0] < total: return 'Bob'
        
        return 'Tie'
    
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        A = stoneValue
        N, total = len(A), sum(A)
        # Becuz only 3 states are used, we can use O(1) space
        # New dp[i] will update dp[i % 3] and use old dp[i + 1]
        # and dp[i + 1 % 3], dp[i + 2 / 3]
        dp = [0] * 3
        
        prefix_sum = 0
        for i in range(N - 1, -1, -1):
            prefix_sum += A[i]
            
            # Cannot use dp[i] = max(dp[i], ...), cuz dp[i] now is value of prev
            # rather the default value for i
            dp[i % 3] = max([prefix_sum - dp[(i + k) % 3] for k in (1, 2, 3)])
        
        return 'Alice' if 2 * dp[0] > total else 'Bob' if 2 * dp[0] < total else 'Tie'
```
