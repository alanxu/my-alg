# Sliding Window

Given an array, the question is usually to get optimal or all answer for a subarray that matching
a condition.

The solution is typically to use 2 moving pointers to mark the left and right of the window, and
keep a running state of the current window. When the cur state match the condition, we update
the ans with state. Return ans after both pointer cannot move any more.

Some common patterns used in the alg:
* Both pointers never move backward
* Running state is updated based on pre state to improve performance
* There are usually two level loops, outter is moving right one by one, inner is moving left when
a condition matches
* Current state will be updated when any pointer moves
* When moving left, usually use left <= right

Types of Sliding Window:
* Type 1: Both left and right needs to move to locate matching cadidates. 
  * Always moving right at outter loop
  * For same right, keep moving left when the condition NOT MATCH
    and you need to find some other condition to move left 
  * Check ans after left stop moving with cur right, ususally
    you need to check if condition match
  * When left pointer moves, the cur state is updated before
  * The state changed based on right pointer can be updated at the beginning or end of outter loop
    
* Type 2: A variable of type 1, excpet the window len is fixed

* Type 3: Moving right pointer to match condition, and left pointer to get optimal ans
  * Always moving right at outter loop
  * For same right, keep moving left when the condition MATCH
  * Check ans BEFORE moving left with cur right
  * When left pointer moves, the cur state is updated before
  * The state changed based on right pointer can be updated at the beginning or end of outter loop



The optimal point is how left can efficiently move

Sliding window algorithms can be implemented with a single pointer and a variable for window size. Typically we use all of the elements within the window for the problem (for eg - sum of all elements in the window).

Two pointer technique is quite similar but we usually compare the value at the two pointers instead of all the elements between the pointers.

Two pointers can also have variations like fast-slow pointer.

## Type 1

### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

```python
# Fav
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        mp = {}
        
        i, ans = 0, 0
        for j in range(len(s)):
            if s[j] in mp:
                # Use max to make sure previous duplication is ignored.
                i = max(mp.get(s[j], 0), i)
            ans = max(ans, j - i + 1)
            mp[s[j]] = j + 1
        return ans
    def lengthOfLongestSubstring(self, s: str) -> int:
        # Intuition: Sliding window
        # Track last pos of a char, if char at right occurs
        # before and last pos is within cur window, move left
        # to that pos + 1
        
        # Condition: subarray with at most k 0s
        # Optimal ans: max len of the subarray
        # Type: The matching candidates have to be found
        # by moving both left and right, so this is type1.
        # Left is moved when condition not match. Optimal
        # ans is checked after moving left.
        left, ans = 0, 0
        # Trick: Use last_pos to check duplicates
        last_pos = {}
        for right in range(len(s)):
            # Only check char at right because others has been checked,
            # needs to check the duplicates is before left
            if s[right] in last_pos and last_pos[s[right]] >= left:
                # Rather than +1, put left directly before duplicated right
                # this is why not use while
                left = last_pos[s[right]] + 1
            ans = max(ans, right - left + 1)
            
            # Update of current window state at the end rather than 
            # beginning
            last_pos[s[right]] = right
        return ans
```

### [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)

```python
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        left = 0
        counts_0 = 0
        ans = 0
        for right, a in enumerate(A):
            if a == 0:
                counts_0 += 1
            while counts_0 > K and left <= right:
                if A[left] == 0:
                    counts_0 -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans
```

### [340. Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)
```python
from collections import defaultdict
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        n = len(s)
        if n * k == 0:
            return 0

        # sliding window left and right pointers
        left, right = 0, 0
        # hashmap character -> its rightmost position
        # in the sliding window
        hashmap = defaultdict()

        max_len = 1

        while right < n:
            # add new character and move right pointer
            hashmap[s[right]] = right
            right += 1

            if len(hashmap) == k + 1:
                # delete the leftmost character
                del_idx = min(hashmap.values())
                del hashmap[s[del_idx]]
                # move left pointer of the slidewindow
                left = del_idx + 1

            max_len = max(max_len, right - left)

        return max_len
        
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        N = len(s)
        if N * k == 0:
            return 0
        left, ans = 0, 0
        counter = defaultdict(int)
        for right in range(N):
            counter[s[right]] += 1
            while len(counter) > k and left < right:
                counter[s[left]] -= 1
                if not counter[s[left]]:
                    del counter[s[left]]
                left += 1

            ans = max(ans, right - left + 1)
        return ans 
                
```

### [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

```python
class Solution:
    def characterReplacement(self, s, k):
        # Trick: Sliding Window
        left, ans = 0, 0
        count = collections.Counter()
        
        for right in range(len(s)):
            count[s[right]] += 1
            max_ = max(count.values())
            # For each step after right increase, kept adjusting left util secific condition
            # is match/not match. Usually left move when the current situation for sure
            # cannot match the answer.
            while right - left + 1 > max_ + k:
                count[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        
        return ans
```

### [1151. Minimum Swaps to Group All 1's Together](https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/)

```python
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        # Trick: Number of 1's in binary array
        ones = sum(data)
        ans = cur_ones = 0
        left = 0
        for right in range(len(data)):
            cur_ones += data[right]
            if right - left + 1 > ones:
                cur_ones -= data[left]
                left += 1
            ans = max(ans, cur_ones)
            
        return ones - ans
```

### [159. Longest Substring with At Most Two Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/)

```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        N, ans = len(s), 0
        if N < 3:
            return N

        left = 0
        for right in range(N):
            while len(set(s[left:right + 1])) > 2:
                left += 1
            ans = max(ans, len(s[left:right + 1]))
        return ans
```

## Type 2

### [30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        N, L, K = len(s), len(words), len(words[0]),
        target_dict = Counter(words)
        ans = []
        for left in range(N - L * K + 1):
            cur_dict = Counter()
            for i in range(L):
                cur_dict[s[left + i * K:left + (i + 1) * K]] += 1
            if cur_dict == target_dict:
                ans.append(left)
        return ans
```

### [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # Trick: Use array of 26 for lengh of each char
        A = [ord(ch) - ord('a') for ch in s1]
        B = [ord(ch) - ord('a') for ch in s2]
        target, window = [0] * 26, [0] * 26
        for x in A:
            target[x] += 1
        
        # Trick: Sliding Window
        # When the window length is fixed, use one moving idx
        for i, x in enumerate(B):
            # Always add new ch
            window[x] += 1
            # When exceed target lenght, always remove last ch
            if i >= len(s1):
                window[B[i - len(A)]] -= 1
            if window == target:
                return True
        
        return False

    def checkInclusion(self, s1: str, s2: str) -> bool:
        N, L = len(s2), len(s1)
        target_dict, cur_dict = Counter(s1), Counter()
        left = 0
        for right in range(N):
            cur_dict[s2[right]] += 1
            if right >= L:
                # left is the prev left, so not right - L + 1
                left = right - L
                cur_dict[s2[left]] -= 1
                if cur_dict[s2[left]] == 0:
                    del cur_dict[s2[left]]
            
            if cur_dict == target_dict:
                return True
        return False
```

## Type 3

### [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        
        l = len(nums)
        sums = [0] * l
        sums[0] = nums[0]
        
        for i in range(1, l):
            sums[i] = sums[i - 1] + nums[i]
        
        ans = float('inf')
        
        for i in range(0, l):
            if sums[i] >= s:
                ans = min(ans, i + 1)
            to_find = s + sums[i]
            
            left, right = i + 1, l - 1
            while left <= right:
                mid = (left + right) // 2
                if sums[mid] >= to_find:
                    ans = min(ans, mid - i)
                    right = mid - 1
                else:
                    left = mid + 1
                    
        return ans if ans != float('inf') else 0
    
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        N = len(nums)
        cur_sum, ans = 0, math.inf
        left = 0
        for right in range(N):
            cur_sum += nums[right]
            
            while left <= right and cur_sum >= target:
                ans = min(ans, right - left + 1)
                cur_sum -= nums[left]
                left += 1
        return ans if ans != math.inf else 0
```

### [1297. Maximum Number of Occurrences of a Substring](https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/)

```python
class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
        # Intuition: maxSize is not required as results matching maxSize have to
        # match minSize.
        res, occ, n = 0, collections.defaultdict(int), len(s)
        for i in range(n - minSize + 1): 
            sub = s[i:i + minSize]
            if len(set(sub)) <= maxLetters:
                occ[sub] += 1
                res = max(res, occ[sub])
        return res
```

### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # Variables needed to check the validity of a window
        t_dict = Counter(t)
        w_dict = {}
        required = len(t_dict)
        formed = 0
        
        # ans tuple of the form (window length, left, right)
        ans = float('inf'), None, None
        
        # Pointers of the window, starting from 0, moving to same direction
        l = r = 0
        
        while r < len(s):
            
            # Update window status according to new right
            c = s[r]
            if c in t_dict:
                w_dict[c] = w_dict.get(c, 0) + 1
                if w_dict[c] == t_dict[c]:
                    formed += 1
                    
            while l <= r and formed == required:
                # When formed == required, record the min window first
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                    
                # Move the left to the position that the window not valid again
                c = s[l]
                if c in t_dict:
                    w_dict[c] -= 1
                    if w_dict[c] < t_dict[c]:
                        formed -= 1
                l += 1
            r += 1
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]
```



## Others



















### [524. Longest Word in Dictionary through Deleting](https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/)

```python
class Solution:
    def findLongestWord(self, s: str, d: List[str]) -> str:
        # For each work in the list, keep an pointer
        p = [0] * len(d)
        
        # 
        matched = []
        for i in range(len(s)):
            for j in range(len(d)):
                # Get pointer for a single target word
                _p = p[j]
                if _p < len(d[j]) and s[i] == d[j][_p]:
                    p[j] += 1
                    if p[j] == len(d[j]):
                        matched.append(d[j])
                        
        matched.sort(key=lambda w: (-len(w), w))

        return matched[0]  if matched else ""   
```

### [532. K-diff Pairs in an Array](https://leetcode.com/problems/k-diff-pairs-in-an-array/)

```python
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        
        nums.sort()
        
        left, right = 0, 1
        result = 0
        
        while left < len(nums) and right < len(nums):
            if left == right or nums[right] - nums[left] < k:
                right += 1
            elif nums[right] - nums[left] > k:
                left += 1
            else:
                result += 1
                left += 1
                while left < len(nums) and nums[left] == nums[left -1]:
                    left += 1
                    
        return result
```



### [1423. Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/)

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        # Find the subarray of length len(cardPoints) - k with minimum sum
        cards = cardPoints
        l  = len(cards)
        # Trick
        _k = l - k
        
        # Create an array of prefix sum
        s = [0] * (l + 1)
        
        for i in range(1, len(s)):
            s[i] = s[i- 1] + cards[i - 1]
            
        ans = float('inf')
        for i in range(_k, len(s)):
            ans = min(ans, s[i] - s[i - _k])
            
        return s[-1] - ans
```

