
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
                # bisect.bisect() returns index of smallest num >= target
                i = bisect.bisect_left(pres, target) - 1
                # If i == -1, means there is no previous nums less than
                # target, that is no sum less than k
                if i >= 0:
                    ans = max(ans, pres[i] + x)
            bisect.insort_left(pres, x)
        return ans
```

### [1679. Max Number of K-Sum Pairs](https://leetcode.com/problems/max-number-of-k-sum-pairs/)

```python
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        mp = defaultdict(int)
        ans = 0
        for i, x in enumerate(nums):
            target = k - x
            if mp[target] > 0:
                mp[target] -= 1
                ans += 1
            else:
                mp[x] += 1
        return ans
```


## Max Size Subarray Sum Equals K

```
525.Contiguous-Array (M)
930.Binary-Subarrays-With-Sum (M)
1442.Count-Triplets-That-Can-Form-Two-Arrays-of-Equal-XOR (H-)
1524.Number-of-Sub-arrays-With-Odd-Sum (M)
974.Subarray-Sums-Divisible-by-K (M)
1590.Make-Sum-Divisible-by-P (M+)
1658.Minimum-Operations-to-Reduce-X-to-Zero (M)
1371.Find-the-Longest-Substring-Containing-Vowels-in-Even-Counts (H-)
1542.Find-Longest-Awesome-Substring (H-)
1915.Number-of-Wonderful-Substrings (M+)
1983.Widest-Pair-of-Indices-With-Equal-Range-Sum (M+)
2025.Maximum-Number-of-Ways-to-Partition-an-Array (H)
```

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

### [1542. Find Longest Awesome Substring](https://leetcode.com/problems/find-longest-awesome-substring/)

```python
class Solution:
    def longestAwesome(self, s: str) -> int:
        """
        Intuition: Hashmap + Prefix + Bitmask
        For each char in s, calc running state of even/odd using
        a mask. The mask indicates each char is of odd or even number.
        Use a hashmap pos to save the ending index of state. For a
        current running state at i, if same state exists with ending at j,
        it means [i,j] has odd num of every chars. Since we allow 1 even,
        so we update current mask state with 1 even char for every char,
        in this case [i, j] is with one more that char, cuz that char become
        diff b/w [i, j].
        Track the max ans for each case for each char.
        """
        # 1 << 10 == 2 ** 10 == 1024
        pos = [math.inf] * 1024
        pos[0] = -1
        
        mask = 0
        ans = 0
        
        for i, x in enumerate(s):
            idx = ord(x) - ord('0')
            mask ^= 1 << idx
            ans = max(ans, i - pos[mask])
            for j in range(10):
                look = mask ^ (1 << j)
                ans = max(ans, i - pos[look])
                
            pos[mask] = min(pos[mask], i)
            
        return ans
```

### [1915. Number of Wonderful Substrings](https://leetcode.com/problems/number-of-wonderful-substrings/)

```python
class Solution:
    def wonderfulSubstrings(self, word: str) -> int:
        counts = [0] * (1 << 10)
        counts[0] = 1
        mask = 0
        ans = 0
        
        for x in word:
            idx = ord(x) - ord('a')
            mask ^= 1 << idx
            ans += counts[mask]
            
            for i in range(10):
                look = mask ^ (1 << i)
                ans += counts[look]
                
            counts[mask] += 1
        
        return ans
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

### [1429. First Unique Number](https://leetcode.com/problems/first-unique-number/)

```python
class FirstUnique:

    def __init__(self, nums: List[int]):
        self.q = deque()
        self.mp = {}
        for x in nums:
            self.add(x)

    def showFirstUnique(self) -> int:
        while self.q and self.mp[self.q[0]]:
            self.q.popleft()
        return self.q[0] if self.q else -1

    def add(self, value: int) -> None:
        if value not in self.mp:
            self.mp[value] = False
            self.q.append(value)
        else:
            self.mp[value] = True


# Your FirstUnique object will be instantiated and called as such:
# obj = FirstUnique(nums)
# param_1 = obj.showFirstUnique()
# obj.add(value)
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
```python
class Solution:
    def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
        by_user_mp, by_pattern_map = defaultdict(list), defaultdict(int)
        for t, u, w in sorted(zip(timestamp, username, website)):
            by_user_mp[u].append(w)
        
        ans, max_score = None, -math.inf
        for k, v in by_user_mp.items():
            # Deduplicate same pattern for same user
            for pattern in set(itertools.combinations(v, 3)):
                by_pattern_map[pattern] += 1
                if max_score < by_pattern_map[pattern] or \
                max_score == by_pattern_map[pattern] and (not ans or pattern < ans):
                    max_score = by_pattern_map[pattern]
                    ans = pattern

        return ans
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