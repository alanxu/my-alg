# Sliding Window

Right always moving forward, left is moved once the limit exceeds.

The optimal point is how left can efficiently move

Sliding window algorithms can be implemented with a single pointer and a variable for window size. Typically we use all of the elements within the window for the problem (for eg - sum of all elements in the window).

Two pointer technique is quite similar but we usually compare the value at the two pointers instead of all the elements between the pointers.

Two pointers can also have variations like fast-slow pointer.

## Problems

### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

```python
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

### [1297. Maximum Number of Occurrences of a Substring](https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/)

```python
class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int: 
        res, occ, n = 0, collections.defaultdict(int), len(s)
        for i in range(n - minSize + 1): 
            sub = s[i:i + minSize]
            if len(set(sub)) <= maxLetters:
                occ[sub] += 1
                res = max(res, occ[sub])
        return res
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

