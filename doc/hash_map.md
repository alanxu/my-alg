
## Two Sum

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


## Max Size Subarray Sum Equals K

### [325. Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)

```
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

