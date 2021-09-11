
## Two Sum

### [1. Two Sum](https://leetcode.com/problems/two-sum/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # It requires returning index, so sorting
        # will mess up index, need index map
        # defaultdict(list) is to handle two same values
        idx_map = defaultdict(list)
        for i, x in enumerate(nums):
            idx_map[x].append(i)

        nums.sort()
        left, right = 0, len(nums) - 1
        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                return [idx_map[nums[left]].pop(), idx_map[nums[right]].pop()]
            elif s < target:
                left += 1
            else:
                right -= 1
    
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mp = {}
        for i, x in enumerate(nums):
            if target - x in mp:
                return [mp[target-x], i]
            mp[x] = i
```

###[1010. Pairs of Songs With Total Durations Divisible by 60](https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/)

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        mp, ans = defaultdict(int), 0
        for i, t in enumerate(time):
            k = 60 - t % 60
            ans += mp[k]
            mp[t % 60] += 1
        return ans + mp[0] * (mp[0] - 1) // 2

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        mp, ans = defaultdict(int), 0
        for t in time:
            # Trick: One more % 60 to fix time dividable by 60
            target = (60 - t % 60) % 60
            ans += mp[target]
            mp[t % 60] += 1
        return ans
```

### [15. 3Sum](https://leetcode.com/problems/3sum/)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res, dups = set(), set()
        seen = {}
        
        for i, v1 in enumerate(nums):
            if v1 in dups:
                continue
            dups.add(v1)
            
            for j, v2 in enumerate(nums[i + 1:]):
                complement = -v1 - v2
                if complement in seen and seen[complement] == i:
                    res.add(tuple(sorted((v1, v2, complement))))
                seen[v2] = i
        return res
```

### [1099. Two Sum Less Than K](https://leetcode.com/problems/two-sum-less-than-k/)

```python
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        pres = []
        ans = -1
        for x in nums:
            target = k - x
            if pres:
                i = bisect.bisect_left(pres, target) - 1
                if i >= 0:
                    ans = max(ans, pres[i] + x)

            bisect.insort_left(pres, x)
            
        return ans
```

## Max Size Subarray Sum Equals K

### [325. Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        # Pattern: Subarray sum of K
        # In this case the answer is able max length, the presence
        # of the sum key matters, so unlike the max nums we just need 
        # to count the num where 0 is ok
        cur_sum, ans = 0, 0
        # Required for [0,i] sums to k, -1 is to offset the 0 index, cuz o is
        # before index 0, so it is -1
        mp = {0: -1}
        for i, x in enumerate(nums):
            cur_sum += x
            
            if cur_sum - k in mp:
                ans = max(ans, i - mp[cur_sum - k])
            
            mp[cur_sum] = mp.get(cur_sum, i)
        
        return ans
```

### [1658. Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/)

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # DP is O(xN^2), not optimal
        # Intuition: Equals to problem - find max len subarray in nums
        # with sum to sum(nums) - x
        
        sum_, N = sum(nums), len(nums)
        if sum_ == x:
            return N
        k = sum_ - x
        cur_sum, ans = 0, 0
        mp = {0: -1}
        for i, v in enumerate(nums):
            cur_sum += v
            if cur_sum - k in mp:
                ans = max(ans, i - mp[cur_sum - k])
            mp[cur_sum] = mp.get(cur_sum, i)
        
        return (N - ans) if ans else -1
```

## Others

### [454. 4Sum II](https://leetcode.com/problems/4sum-ii/)
```python
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        lists = [A, B, C, D]
        Len = len(lists)
        map = {}
        
        # Trick: K sum problem
        #        Use hashmap
        def process_first_grp(i=0, sum=0):
            if i == Len // 2:
                map[sum] = map.get(sum, 0) + 1
            else:
                for num in lists[i]:
                    process_first_grp(i + 1, sum + num)
        
        def process_second_group(i=Len // 2, sum=0):
            if i == Len:
                return map.get(-sum, 0)
            else:
                count = 0
                for num in lists[i]:
                    count += process_second_group(i + 1, sum + num)
                return count
        
        process_first_grp()
        return process_second_group()
```

### [1604. Alert Using Same Key-Card Three or More Times in a One Hour Period](https://leetcode.com/problems/alert-using-same-key-card-three-or-more-times-in-a-one-hour-period/)

```python
class Solution:
    def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
        mapping = defaultdict(list)
        for name, time in zip(keyName, keyTime):
            hour, minu = map(int, time.split(":"))
            time = hour * 60 + minu
            mapping[name].append(time)
        
        ans = []
        for name, times in mapping.items():
            # handle case not single day [23:59, 00:01] 
            times.sort()
            q = deque()
            for time in times:
                q.appendleft(time)
                while q[0] - q[-1] > 60:
                    q.pop()
                if len(q) >= 3:
                    ans.append(name)
                    break
        return sorted(ans)
```

### [36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)

### [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

```python
class Solution(object):
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            # Trick: sorted() can be used 
            # on str
            ans[tuple(sorted(s))].append(s)
        return ans.values()
```

### [1152. Analyze User Website Visit Pattern](https://leetcode.com/problems/analyze-user-website-visit-pattern/)

```python
class Solution:
    def mostVisitedPattern(self, username, timestamp, website):
        dp = collections.defaultdict(list)
        for t, u, w in sorted(zip(timestamp, username, website)):
            dp[u].append(w)
        count = sum([collections.Counter(set(itertools.combinations(dp[u], 3))) for u in dp], collections.Counter())
        return list(min(count, key=lambda k: (-count[k], k)))
```

### [670. Maximum Swap](https://leetcode.com/problems/maximum-swap/)

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        # Inuition: Greedy, Hashmap
        # Create a map with key = <each digit in num>,
        # and value = <last index of that value>. For 
        # each i, x in num, find (j, y) after i where 
        # y is the max in num[i + 1:], if there are more
        # than 1 of y, swap the last one. Why last one?
        # 27376 -> 77326
        # 27376 -> 72376
        # Swap with last max always yields greater value.
        A = list(map(int, str(num)))
        last_index = { v: i for i, v in enumerate(A)}
        # Iterate from start of num and return as long
        # as there is a greater value found for cur i.
        # Bcuz i is the most significant digit
        for i, v1 in enumerate(A):
            # Trick: To find the max value after i, start
            # from 9 until v1 + 1, look up the index j
            # in the map. This recude the search for
            # max to O(1), whilst if loop for all elements
            # after i, it takes O(n)
            for v2 in range(9, v1, -1):
                j = last_index.get(v2, -1)
                if j > i:
                    A[i], A[j] = A[j], A[i]
                    return int(''.join(map(str, A)))

        return num
```

### [205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        # Intuition: Compare if the two words follow same pattern.
        # No worries if same charactor in diff position in diff str.
        # We just care the pattern.
        # We sign id to each char in both str based on position. The
        # id is the pos of the first occurance of a char
        # paper: 01034
        # title: 01034
        # Simply compare counts of each char cannot work, cuz it doesn't
        # compare position.
        map_a, map_b = {}, {}
        for (i, a), (j, b) in zip(enumerate(s), enumerate(t)):
            if a in map_a and b in map_b:
                # If both happens before, check the id this char are same
                if map_a[a] != map_b[b]:
                    return False
            elif a not in map_a and b not in map_b:
                # If both not happen before, set id
                map_a[a], map_b[b] = i, j
            else:
                # If one happends one didn't happen, return False
                return False
            
        return True
```

### [389. Find the Difference](https://leetcode.com/problems/find-the-difference/)

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        counter_s, counter_t = collections.Counter(s), collections.Counter(t)
        for c in counter_t.keys():
            if c not in counter_s or counter_s[c] < counter_t[c]:
                return c
```


### [1331. Rank Transform of an Array](https://leetcode.com/problems/rank-transform-of-an-array/)

```python
class Solution:
    def arrayRankTransform(self, arr: List[int]) -> List[int]:
        idx = {x: i + 1 for i, x in enumerate(sorted(set(arr)))}
        return [idx[i] for i in arr]
```