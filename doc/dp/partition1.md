### [1278. Palindrome Partitioning III](https://leetcode.com/problems/palindrome-partitioning-iii/)

```python
class Solution:
    def palindromePartition(self, s: str, K: int) -> int:
        # Pattern: DP - Partition 1
        # There are fixed partitions
        N = len(s)
        # Trick: For 2D dp, add 0 to each dimension, sometimes only
        # dp[0][0] has valid value, others just inf. This works for
        # cases use min(), max()
        # Inutition: dp[i][k] is min replacement to make [0, i] palindrome
        dp = [[math.inf] * (K + 1) for _ in range(N + 1)]
        dp[0][0] = 0
        
        # Pattern: Palindrome with replacement is relatively easy
        def num_of_changes(i, j):
            ans = 0
            while i < j:
                if s[i] != s[j]:
                    ans += 1
                i, j = i + 1, j - 1
            return ans
        
        for i in range(1, N + 1):
            for k in range(1, min(K, i) + 1):
                # Iterate a possible j, where [0, j - 1] forms k - 1 partitions
                # and [j, i] firms k-th partition
                for j in range(k, i + 1):
                    dp[i][k] = min(dp[i][k], dp[j - 1][k - 1] + num_of_changes(j - 1, i - 1))
        
        return dp[N][K]
```

### [813. Largest Sum of Averages](https://leetcode.com/problems/largest-sum-of-averages/)

```python
class Solution:
    def largestSumOfAverages(self, A: List[int], K: int) -> float:
        # Pattern: DP - Partition 1
        # Inuition: When the question asks about with maximum k partition
        # you just find out [0, k] partition and use max in result. No
        # difference than as exact k partitions
        N = len(A)
        dp = [[-math.inf] * (K + 1) for _ in range(N + 1)]
        dp[0][0] = 0
        
        for i in range(1, N + 1):
            for k in range(1, min(i, K) + 1):
                for j in range(k, i + 1):
                    # Always be careful about i - 1 vs i when add dummy rows in dp
                    dp[i][k] = max(dp[i][k], dp[j - 1][k - 1] + sum(A[j - 1:i]) / (i - j + 1))
        
        return max(dp[-1])
    
    def largestSumOfAverages(self, A: List[int], K: int) -> float:
        N = len(A)
        dp = [[-math.inf] * (K + 1) for _ in range(N + 1)]
        dp[0][0] = 0
        
        for i in range(1, N + 1):
            for k in range(1, min(i, K) + 1):
                prefix_sum = 0
                # The order of j can be either, cuz it just depends on prev level of
                # of i ([j - 1][k - 1]), not in current round
                for j in range(i, k - 1, -1):
                    prefix_sum += A[j - 1]
                    dp[i][k] = max(dp[i][k], dp[j - 1][k - 1] + prefix_sum / (i - j + 1))
        
        return max(dp[-1])
```

### [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)

```python
class Solution:
    def splitArray(self, nums: List[int], K: int) -> int:
        # TLE
        N = len(nums)
        dp = [[math.inf] * (K + 1) for _ in range(N + 1)]
        dp[0][0] = 0
        
        for i in range(1, N + 1):
            for k in range(1, min(K, i) + 1):
                dp[i][k] = math.inf
                prefix_sum = 0
                for j in range(i, k - 1, -1):
                    # When k is 1, j has to be 1, but we still do iteration,
                    # because when k == 0, dp[i][k] = inf except when i = 0,
                    # so the interation is not inf only when j = 1 (first element)
                    prefix_sum += nums[j - 1]
                    dp[i][k] = min(dp[i][k], max(dp[j - 1][k - 1], prefix_sum))
        
        return dp[-1][-1]
        
    def splitArray(self, nums: List[int], K: int) -> int:
        # Alg: Binary Search
        # Intuition: Search for min max sum of one partition, left is max of nums, right
        # is sum of nums. The lager of the ans, the fewer the partions. The answer is
        # the min value can keep <= K partitions, decrease by 1, the partitions will > K.
        # So it is the edge value that can be found by binary search.
        # The edge case is gauanteed to make l and r meet.
        N = len(nums)
        l = r = 0
        
        for x in nums:
            r += x
            l = x if x > l else l
        
        ans = r
        while l < r:
            mid = (l + r) // 2
            
            sum_, counts = 0, 1
            for x in nums:
                sum_ += x
                if sum_ > mid:
                    counts += 1
                    sum_ = x
            
            if counts <= K:
                ans = min(ans, mid)
                r = mid
            else:
                l = mid + 1
        
        return ans
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

### [1478. Allocate Mailboxes](https://leetcode.com/problems/allocate-mailboxes/)

```python
class Solution:
    def minDistance(self, houses: List[int], K: int) -> int:
        N = len(houses)
        dp = [[math.inf] * (K + 1) for _ in range(N)]
        houses.sort()
        
        costs = [[0] * N for _ in range(N)]
        
        # Trick: Manhaton Distance - Put it in median item
        for i, j in product(range(N), range(N)):
            median = houses[(i + j) // 2]
            for t in range(i, j + 1):
                costs[i][j] += abs(median - houses[t])
        
        for i in range(N):
            # Trick: Avoid using if k == 1
            dp[i][1] = costs[0][i]
            # What if k is more than i? Just interrate it.
            # When i == 1, for loop of j will not run, so
            # dp[1][2...] all inf, which actually should be 0 but
            # doesn't matter.
            # For k more than i, dp[i][k] == 0
            for k in range(2, K + 1):
                # Iterate on j can make sure find optimal division for the new
                # mailbox
                for j in range(1, i + 1):
                    dp[i][k] = min(dp[i][k], dp[j - 1][k - 1] + costs[j][i])
        print(dp)
        return dp[-1][-1]
```

### [629. K Inverse Pairs Array](https://leetcode.com/problems/k-inverse-pairs-array/)

```python
class Solution:
    def kInversePairs(self, N: int, K: int) -> int:
        # Pattern: DP - Patition 1
        # Given a sequence (i) and a independent constraints (k), partition, mailbox
        # num of inversed seq etc. Ask for a optimal answer.
        # Usually iterate for i and k, for dp[i][k], iterate all cases specific for
        # A[i] and aggregate it for dp[i][k]
        
        # Intuition: dp[i][k] is to add A[i] into [0, i - 1] which has dp[i-1][...],
        # There are i + 1 ways to put A[i], when put A[i] at i, it will not create
        # extra inversed seq; when put A[i] at 0, it create extra i - 1 inversed seq,
        # bcuz A[i] is max value in [0, i]. 
        # Putting A[i] in [0, i - 1] will only adds to the results but not changing
        # exsting value for [0, i - 1]. So when A[i] generate m cases, [0, i - 1] needs
        # to generate k - m cases. dp[i][k] is the sum of the two parts.
        # The cases putting A[i] can generate is bounded by i and k, choose the min.
        
        # This alg will TLE, needs to use fomular for 3rd loop.
        dp = [[0] * (K + 1) for _ in range(N + 1)]
        
        for i in range(N + 1):
            dp[i][0] = 1
        
        for i in range(1, N + 1):
            for k in range(1, K + 1):
                if k == 0:
                    dp[i][k] = 1
                else:
                    for m in range(min(k, i - 1) + 1):
                        dp[i][k] = (dp[i][k] + dp[i - 1][k - m]) % 1000000007
        return dp[-1][-1]

    def kInversePairs(self, N: int, K: int) -> int:
        dp = [[0] * (K + 1) for _ in range(N + 1)]
        
        for i in range(N + 1):
            dp[i][0] = 1
        
        for i in range(1, N + 1):
            for k in range(1, K + 1):
                if k >= i:
                    dp[i][k] = dp[i][k - 1] + dp[i - 1][k] - dp[i - 1][k - i]
                else:
                    dp[i][k] = dp[i][k - 1] + dp[i - 1][k]
                dp[i][k] %= 10 ** 9 + 7
        
        return dp[-1][-1]
```

### [1621. Number of Sets of K Non-Overlapping Line Segments](https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/)

```python

```