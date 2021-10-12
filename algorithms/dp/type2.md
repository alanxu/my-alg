# Description

1D DP
DP[i] depends on some previous steps


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

## Word Break


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

### [472. Concatenated Words](https://leetcode.com/problems/concatenated-words/)

```python
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        # Pattern: Top-down DP
        # Trick: Convert list into hashset for O(1) search
        words = set(words)
        @functools.lru_cache(None)
        def is_qualified(word, is_partial=False):
            # Use is_partial to mark calls for sub problem, if it is partial
            # and in words, it means the original word is qualified
            if is_partial and word in words:
                return True
            # Iteratively split word into two parts, first part is in words,
            # second parts for another recursive call
            for i in range(1, len(word)):
                if word[:i] in words and is_qualified(word[i:], True):
                    return True

        ans = []
        for w in words:
            if is_qualified(w):
                ans.append(w)
        return ans
```
```python
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        word_set = set(words)
        @functools.lru_cache()
        def dfs(word, partial=False):
            # Termination condision is word itself in list and it is not
            # the top level word
            if partial and word in word_set:
                return True
            
            # Get substring not the entire word (last char is skipped)
            # this will not check the word itself.
            for i in range(1, len(word)):
                if word[:i] in word_set and dfs(word[i:], True):
                    return True

            return False
        
        ans = []
        for w in words:
            if dfs(w):
                ans.append(w)
        
        return ans
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
            # it shold be len(s) + 1, because range()'s end is exclusive
            for end in range(1, len(s) + 1):
                word = s[:end]
                if word in words:
                    for sublist in helper(s[end:]):
                        ans.append([word] + sublist)
            return ans
        
        return [' '.join(words) for words in helper(s)]
```

### [1048. Longest String Chain](https://leetcode.com/problems/longest-string-chain/)
```python
class Solution:
    def longestStrChain1(self, words: List[str]) -> int:
        # Trick: DP - Bottom up
        # Trick: Use dict for DP, cus the key is str this time
        dp = {}
        # Trick: Sort the input by len
        # Trick: Use len as support point, words with same length for sure not in same chain
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
    
    def longestStrChain(self, words: List[str]) -> int:
        # DP - Type 2
        N = len(words)
        # Trick: Sort by len
        words.sort(key=len)
        indexs = {x: i for i, x in enumerate(words)}

        # dp[i] is max len of word chain in [0, i]
        dp = [1] * N
        
        for i in range(1, N):
            w = words[i]
            # Interate each i, find some previous j, typical type 2
            for k in range(len(w)):
                j = indexs.get(w[:k] + w[k+1:], -1)
                if j >= 0:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
```


### [446. Arithmetic Slices II - Subsequence](https://leetcode.com/problems/arithmetic-slices-ii-subsequence/)

```python
class Solution:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        N = len(A)
        # dp[i] is a map for A[i], key is d, value is num of
        # arithmetic subseq ending with i with difference d, regardless
        # if length
        dp = [defaultdict(int) for _ in range(N)]
        
        ans = 0
        for i in range(1, N):
            for j in range(i):
                diff = A[i] - A[j]
                local_counts = dp[j][diff]
                # Accumulate accounts of arithmetic subseqs ending with
                # i: increased by local_counts means all counts for i plus
                # i forms same num of seqs for i, then (j, i) is other 1
                dp[i][diff] += local_counts + 1
                # We count local_counts as the new increased valid number,
                # because the extra 1 is length of 2. The seq for i might
                # be just 2 items, add i forms 3 items, so just simply add
                # local_counts. If i has no seq ending with j, local_accounts
                # == 0, so just smply add it.
                ans += local_counts
        return ans
```

### [1027. Longest Arithmetic Subsequence](https://leetcode.com/problems/longest-arithmetic-subsequence/)

```python
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = [defaultdict(int) for _ in range(N)]
        ans = 0
        for i in range(1, N):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = dp[j][diff] + 1
                ans = max(ans, dp[i][diff] + 1)
        return ans
    
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = {}
        for i in range(1, N):
            for j in range(i):
                diff = A[i] - A[j]
                dp[(diff, i)] = dp.get((diff, j), 1) + 1
        
        return max(dp.values())
```


## Others

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


### [368. Largest Divisible Subset](https://leetcode.com/problems/largest-divisible-subset/)

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        # The inuition: if a % b = 0 and b % c = 0, then a % c = 0
        
        # Trick: Sort the array for combination questions to reduce times of iteration
        nums = sorted(nums)
        
        # Dict key is max num in that subset, value is the divisible subset
        subsets = {-1: set()}
        
        for num in nums:
            subsets[num] = max([subsets[k] for k in subsets if num % k == 0], key=len) | {num}
            
        return max([subsets[k] for k in subsets], key=len)
        
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        # Intuition: DP type 2
        # It is easy if the anwser is length, but it is the any matching sebseq,
        # we can use same method and record the related j, i using prevs, and
        # find the index of the end of max length, and use last idx and the prevs
        # to get all subseq
        N = len(nums)
        # Sort so we just used nums[i] % nums[j] == 0
        nums.sort()
        
        dp = [1] * N
        prevs, last_idx = [-1] * N, 0
        
        for i in range(1, N):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    dp[i] = max(dp[i], dp[j] + 1)
                    if dp[i] == dp[j] + 1:
                        prevs[i] = j
        
        max_idx = dp.index(max(dp))
        ans = []
        while max_idx >= 0:
            ans.append(nums[max_idx])
            max_idx = prevs[max_idx]
                    
        return ans
```

### [1105. Filling Bookcase Shelves](https://leetcode.com/problems/filling-bookcase-shelves/)

```python
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelf_width: int) -> int:
        # Pattern: DP type2
        N, W = len(books), shelf_width
        
        # Intuition: dp[i] is min height to place [0,i - 1] books. Try all possible
        # placement:
        # - book[i - 1] in new line: dp[i] = dp[i - 1] + books[i - 1][1]
        # - book[i - 1] together with each previous books when sum width <= shelf_w:
        #   dp[i] = min(dp[i], dp[j - 1] + height_of_cur_row)
        # It can work, because dp[j - 1] is not in same row as dp[i], so it is a
        # subproblem
        dp = [0] * (N + 1)
        dp[0]= 0
        
        for i in range(1, N + 1):
            width = books[i - 1][0]
            height = books[i - 1][1]
            j = i - 1
            dp[i] = dp[i - 1] + height
            while books[j - 1][0] + width <= W and j >= 1:
                width += books[j - 1][0]
                height = max(height, books[j - 1][1])
                dp[i] = min(dp[i], dp[j - 1] + height)
                j -= 1
        
        return dp[-1]
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

### [132. Palindrome Partitioning II](https://leetcode.com/problems/palindrome-partitioning-ii/)

```python
class Solution:
    def minCut(self, s: str) -> int:
        # Pattern: DP Partition 1 - TLE
        N = len(s)
        dp = [[False] * (N + 1) for _ in range(N)]
        
        def is_valid(s):
            return s == s[::-1]
        
        for i in range(N):
            for k in range(1, min(i + 1, N) + 1):
                if k == 1:
                    dp[i][1] = is_valid(s[:i + 1])
                else:
                    for j in range(k - 1, N):
                        dp[i][k] = dp[i][k] or dp[j - 1][k - 1] and is_valid(s[j:i + 1])
        # print(dp)
        return min(i for i, x in enumerate(dp[-1]) if x) - 1
                        
    def minCut(self, s: str) -> int:
        # Pattern: DP Type 2
        # Intuition: dp[i] is min cut for [0, i] to make palindrome partitions.
        # create a dp with size n + 1, first -1 is for dp[j - 1] when j is first
        # in s.
        N = len(s)
        dp = [-1] + [math.inf] * N
        dp[1] = 0
        
        def is_valid(s):
            return s == s[::-1]
        
        for i in range(1, N + 1):
            for j in range(1, i + 1):
                if s[j - 1] == s[i - 1] and is_valid(s[j - 1: i]):
                    dp[i] = min(dp[i], dp[j - 1] + 1)

        return dp[-1]
```

### [940. Distinct Subsequences II](https://leetcode.com/problems/distinct-subsequences-ii/)

```python
class Solution:
    def distinctSubseqII(self, S: str) -> int:
        # Pattern - DP Type 2
        # Pattern - How many ways
        # - Select and No Select
        # - Consier first or last
        # - Handle duplications
        # Pattern: See 10^9 + 7 know it is DP
        # Trick: You must consier empty. If not, dp[0] = 1 for 'a', for j in [1..n], 
        # you always consider add or not add S[j] based on 'a', which is wrong.
        # You have to have dp[0] = 1 for '', then dp[1] = 2 for 'a', then dp[2] can
        # be just 'b' not always start from 'a'. BUT you need to -1 at the end to 
        # remove the case that all chars are not selected.
        S= ' ' + S
        N = len(S)
        dp = [0] * (N)
        dp[0] = 1
        
        last_idx = {}
        for i in range(1, N):
            # Select S[i] and not select S[i]
            dp[i] = dp[i - 1] + dp[i - 1]
            
            # There will be duplicate seq if S[i] == S[j],
            # Then dp[i - 1] * 1(S[i]) and dp[i - 1] * 1(S[i]).
            # The duplication only happen for j where j is last
            # same value as i.
            # Why last not ealier?
            # X X X a X X a X X
            # X X X   X X a X X a
            # Not same
            # X X X a     a
            # X X X       a     a
            # Same, but this is is counted in last a case??
            j = last_idx.get(S[i])
            if j:
                dp[i] -= dp[j - 1]
            dp[i] %= 10 ** 9 + 7
            
            last_idx[S[i]] = i
                
        return dp[-1] - 1
```


