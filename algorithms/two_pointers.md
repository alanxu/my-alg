## K Sums

### [18. 4Sum](https://leetcode.com/problems/4sum/)

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def k_sum(nums: List[int], target: int, k: int):   
            if not nums or nums[0] * k > target or nums[-1] * k < target:
                return []
            
            if k == 2:
                return two_sum(nums, target)
            
            l = len(nums)
            ret = []
            for i in range(l):
                if i == 0 or nums[i - 1] != nums[i]:
                    for r in k_sum(nums[i + 1:], target - nums[i], k - 1):
                        ret.append(r + [nums[i]])
                        
            return ret
            
        def two_sum(nums, target):
            lo, hi = 0, len(nums) - 1
            ret = []
            while lo < hi:
                s = nums[lo] + nums[hi]
                if s > target or (hi < len(nums) - 1 and nums[hi] == nums[hi + 1]):
                    hi -= 1
                elif s < target or (lo >= 0 and nums[lo] == [lo - 1]):
                    lo += 1
                else:
                    ret.append([nums[lo], nums[hi]])
                    lo += 1
                    hi -= 1
            return ret
        
        nums.sort()
        return k_sum(nums, target, 4)
```

### [3Sum Closest](https://leetcode.com/problems/3sum-closest/)
```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        
        diff = float('inf')
        nums.sort()
        for i in range(len(nums)):
            lo, hi = i + 1, len(nums) - 1
            while lo < hi:
                sum = nums[i] + nums[lo] + nums[hi]
                if abs(target - sum) < abs(diff):
                    diff = target - sum
                if diff == 0:
                    break
                if sum < target:
                    lo += 1
                else:
                    hi -= 1
            
        return target - diff
```

### [259. 3Sum Smaller](https://leetcode.com/problems/3sum-smaller/)

```python
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        nums.sort()
        sum = 0
        for i in range(len(nums)):
            sum += self.twoSumSmaller(nums[i + 1:], target - nums[i])
            
        return sum
            
    def twoSumSmaller(self, nums, target):
        lo, hi = 0, len(nums) - 1
        sum = 0
        
        while lo < hi:
            s = nums[lo] + nums[hi]
            if s < target:
                sum += hi - lo
                lo += 1
            else:
                hi -= 1
        return sum

    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        # Inuition: Sort nums first. For each i, for each left after
        # i, find the right so that 3sum < target, then i, left and [left + 1, right]
        # firms the ans for i and left. Then keep changing i and left and find
        # their right.
        ans = 0
        nums.sort()
        for i in range(len(nums)):
            left, right = i + 1, len(nums) - 1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if s < target:
                    # Trick: Add a range not only 1 tuple
                    ans += right - left
                    # Is there duplication when increment left
                    # and keep looping? No, all counted ans are
                    # based on i and previous left
                    left += 1
                else:
                    right -= 1
        return ans
```

## Read Write Pointers

### [443. String Compression](https://leetcode.com/problems/string-compression/)

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        l = len(chars)
        
        read, write = 0, 0
        
        count = 0
        
        while read < l:
            c = chars[read]
            count += 1
            if read == l - 1 or c != chars[read + 1]:
                chars[write] = c
                write += 1
                if count > 1:
                    for s in str(count):
                        chars[write] = s
                        write += 1
                count = 0
            read += 1
            
        return write
```

### [723. Candy Crush](https://leetcode.com/problems/candy-crush/)

```python
class Solution(object):
    def candyCrush(self, board):
        R, C = len(board), len(board[0])
        bd = board
        todo = False
        
        # Scan and mark adjacent same cell in each row
        for r in range(R):
            # Trick: check adjacent same cell in a matrix
            # Trick: use negative value to mark value in place
            for c in range(C - 2):
                # Use abs and check != 0
                if abs(bd[r][c]) == abs(bd[r][c + 1]) == abs(bd[r][c + 2]) != 0:
                    bd[r][c] = bd[r][c + 1] = bd[r][c + 2] = -abs(bd[r][c])
                    # If there is any candy marked, it means the board needs to
                    # be re-processed again.
                    todo = True
        
        # Scan and mark adjacent same cell in each col
        for r in range(R - 2):
            for c in range(C):
                if abs(bd[r][c]) == abs(bd[r + 1][c]) == abs(bd[r + 2][c]) != 0:
                    bd[r][c] = bd[r + 1][c] = bd[r + 2][c] = -abs(bd[r][c])
                    todo = True
                    
        # Sweep marked candies and rearrange cells for current round
        for c in range(C):
            # Trick: How to 'compact' an array from one end to the other?
            # - Use read/write pointers
            # - Read points to next value to be read, write points to next
            #   value to be writen
            # - Read pointer always moving forward
            # - Write pointer move forward only when a write happen
            write = R - 1
            for read in range(R - 1, -1, -1):
                if bd[read][c] > 0:
                    # There will be some values replaced by themselves
                    bd[write][c] = bd[read][c]
                    write -= 1
            
            # When read reachs to its end, write is pointing to first empty location in the column
            # , so continue moving write and set all remaining to 0
            for write in range(write, -1, -1):
                bd[write][c] = 0
                
        return self.candyCrush(bd) if todo else bd
```

## Shortest Distance

### [821. Shortest Distance to a Character](https://leetcode.com/problems/shortest-distance-to-a-character/)

```python
class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
        c_index = -math.inf
        ans = []
        for i, x in enumerate(s):
            if x == c:
                c_index = i
            ans.append(i - c_index)
        
        c_index = math.inf
        for i in range(len(s) - 1, -1, -1):
            if s[i] == c:
                c_index = i
            ans[i] = min(ans[i], c_index - i)
        
        return ans
```


### [838. Push Dominoes](https://leetcode.com/problems/push-dominoes/)

```python
class Solution:
    def pushDominoes(self, d: str) -> str:
        # Intuition: Whether be pushed or not, depend on the shortest 
        # distance to 'L' and 'R'.
        d = 'L' + d + 'R'
        ans = ""
        i = 0
        for j in range(1, len(d)):
            if d[j] == '.':
                continue
            if i:
                ans += d[i]
            middle = j - i - 1
            if d[i] == d[j]:
                ans += d[i] * middle
            elif d[i] == 'L' and d[j] == 'R':
                ans += '.' * middle
            else:
                ans += 'R' * (middle // 2) + '.' * (middle % 2) + 'L' * (middle // 2)
            i = j
        return ans
```


## Others

### [244. Shortest Word Distance II](https://leetcode.com/problems/shortest-word-distance-ii/)

```python
class WordDistance:

    def __init__(self, words: List[str]):
        self.locations = defaultdict(list)
        for i, w in enumerate(words):
            self.locations[w].append(i)

    def shortest(self, word1: str, word2: str) -> int:
        # Trick: Two Pointers
        #        The answer can be enumerated by two pointers,
        #        and every step, moving one pointer can leads to
        #        a optimal answer.
        loc1, loc2 = self.locations[word1], self.locations[word2]
        i, j = 0, 0
        min_dis = math.inf
        while i < len(loc1) and j < len(loc2):
            min_dis = min(min_dis, abs(loc1[i] - loc2[j]))
            if loc1[i] < loc2[j]:
                i += 1
            else:
                j += 1
        return min_dis
```

### [658. Find K Closest Elements](https://leetcode.com/problems/find-k-closest-elements/)

```python
class Solution:
    def findClosestElements1(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - 1
        while right - left + 1 > k:
            if abs(arr[left] - x) > abs(arr[right] - x):
                left += 1
            else:
                # When left and right have same distance, chosse left
                # |a - x| == |b - x| and a < b
                right -= 1
        return arr[left:right+1]
    
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        arr.sort(key=lambda n: abs(n - x))
        return sorted(arr[:k])
```

### [75. Sort Colors](https://leetcode.com/problems/sort-colors/)

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Dutch National Flag problem solution.
        """
        # for all idx < p0 : nums[idx < p0] = 0
        # curr is an index of element under consideration
        p0 = curr = 0
        # for all idx > p2 : nums[idx > p2] = 2
        p2 = len(nums) - 1

        while curr <= p2:
            if nums[curr] == 0:
                nums[p0], nums[curr] = nums[curr], nums[p0]
                p0 += 1
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[p2] = nums[p2], nums[curr]
                p2 -= 1
            else:
                curr += 1
```

### [1658. Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/)

### [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        
        ans = 0
        while l < r:
            ans = max(ans, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
                
        return ans
```



### [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # Alg: Two Pointers
        # Intuition: Use left, right to point to
        # cur point. Every loop, calculate the lower
        # side, why? when h[left] < h[right], 
        # we know max_left < max_right,
        # because if the max height is at index left,
        # it will stay there until ther new max height
        # happen at right; also, when h[left] < h[right],
        # we know max_left >= h[left] < h[right] <= max_right
        # and given max_left <= max_right, ans for left
        # is out.
        left, right = 0, len(height) - 1
        max_left, max_right = 0, 0
        ans = 0
        while left < right:
            if height[left] < height[right]:
                if height[left] < max_left:
                    ans += max_left - height[left]
                else:
                    max_left = height[left]
                left += 1
            else:
                if height[right] < max_right:
                    ans += max_right - height[right]
                else:
                    max_right = height[right]
                right -= 1
        return ans
```

### [151. Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        left, right = 0, len(s) - 1
        
        while left < right and s[left] == ' ':
            left += 1
            
        while left < right and s[right] == ' ':
            right -= 1
        
        words, word = deque(), []
        while left <= right:
            if s[left] != ' ':
                word.append(s[left])
            elif s[left] == ' ':
                if word:
                    words.appendleft(''.join(word))
                word = []
            left += 1
        words.appendleft(''.join(word))
        
        return ' '.join(words)
```

### [186. Reverse Words in a String II](https://leetcode.com/problems/reverse-words-in-a-string-ii/)

```python
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        N = len(s)
        def reverse(left, right):
            # Trick: Reverse a string in place
            while left < right:
                s[left], s[right] = s[right], s[left]
                left, right = left + 1, right - 1
        
        def reverse_each_word():
            left = 0
            for right in range(N):
                if right == N - 1 or s[right + 1] == ' ':
                    reverse(left, right)
                    left = right + 2
        
        reverse(0, N - 1)
        reverse_each_word()
```

### [763. Partition Labels](https://leetcode.com/problems/partition-labels/)

```python
class Solution:
    def partitionLabels(self, S: str) -> List[int]:            
        last = {c: i for i, c in enumerate(S)}
        
        left, right = 0, 0
        ans = []
        
        for i, c in enumerate(S):
            right = max(right, last[c])
            
            if i == right:
                ans += [right - left + 1]
                left = right + 1
                
        return ans
```

### [1574. Shortest Subarray to be Removed to Make Array Sorted](https://leetcode.com/submissions/detail/452769445/)

```
class Solution {
public:
    int findLengthOfShortestSubarray(vector<int>& A) {
        int N = A.size(), left = 0, right = N - 1;
        while (left + 1 < N && A[left] <= A[left + 1]) ++left;
        if (left == A.size() - 1) return 0;
        while (right > left && A[right - 1] <= A[right]) --right;
        int ans = min(N - left - 1, right), i = 0, j = right;
        while (i <= left && j < N) {
            if (A[j] >= A[i]) {
                ans = min(ans, j - i - 1);
                ++i;
            } else ++j;
        }
        return ans;
    }
};
```

### [68. Text Justification](https://leetcode.com/problems/text-justification/)

```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # Intuition: Use two pointers to 'cut' the words, use a func justify_line()
        # to format each line. 
        def justify_line(start, end, total_len):
            # Use start/end instead of slicing to improve spead and space
            # Use total_len to denote the total len of all words in this line, this
            # is to save time to repeat claculating word lens
            
            # Given len of words in the line, calculate how many spaces required
            space_len = maxWidth - total_len
            if end > start:
                # If more than 1 words in the line, calc average space b/w each word
                # and also total spare spaces needs to spread in front words.
                space_unit, space_offset = divmod(space_len, end - start)
                # Compose output starting from first word
                outputs = [words[start]]
                for i in range(end - start):
                    # Starting from 2nd word, append average len of spaces first
                    outputs.append(' ' * space_unit)
                    # Then spread the offset space, the number of space is guaranteed
                    # to be less than (end - start) i.e. gaps b/w words in the line.
                    # So each gap will have at most 1 offset space.
                    if space_offset:
                        outputs.append(' ')
                        space_offset -= 1
                    # After fill the spaces, add the word
                    outputs.append(words[start + i + 1])
                return ''.join(outputs)
            else:
                # If only one word in the line, returns word followed by spaces.
                return words[start] + ' ' * space_len
        
        N = len(words)
        start = end = 0
        total_len = 0
        ans = []
        # Use two pointers, use end to add new items
        while start <= end and end < N:
            l = len(words[end])
            # Trick: Don't directly update the status, check if cur end should be
            # added to cur line or a new line.
            if total_len + l + end - start > maxWidth:
                # Note to consider num of spaces when compare with maxWidth (end - start)
                # is num of words in cur line minus one, it equals the mininum spaces required
                # If add cur word no longer valid, update cur line
                ans.append(justify_line(start, end - 1, total_len))
                total_len = l
                start = end
            else:
                # If can add to cur line, update cur line and continue looping
                total_len += l
            end += 1
        # Don't forget to process last line, last line requires 1 spaces in b/w, followed by
        # padding spaces.
        last_line = ' '.join(words[start:end])
        ans.append(''.join([last_line, ' ' * (maxWidth - len(last_line))]))
        return ans
```


### [611. Valid Triangle Number](https://leetcode.com/problems/valid-triangle-number/)

```python
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        # TLE, bcos it is O(N^3)
        self.ans, self.N = 0, len(nums)
        def is_valid(a, b, c):
            return a + b > c and a + c > b and b + c > a
        def backtrack(start, triangle):
            if len(triangle) == 3:
                self.ans += 1
                return
            for i in range(start, self.N):
                x = nums[i]
                if len(triangle) < 2 or sum(triangle) > x :
                    backtrack(i + 1, triangle + [x])
                else:
                    break
        nums.sort()
        backtrack(0, [])
        return self.ans
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        N, ans = len(nums), 0
        # Choose 1st num i reversely [N - 1, 2], for each 'end'
        # at the left of it, calculte valid 'start' that can
        # form triange with i and end. If matched, all items
        # [start, end) is valid with i and end. So for i and end,
        # the num of cases are (end - start). If start is valid,
        # [start, end) are valid. Sum up on all i and end.
        for i in range(N - 1, 1, -1):
            start, end = 0, i - 1
            while start < end:
                if nums[start] + nums[end] > nums[i]:
                    ans += end - start
                    end -= 1
                else:
                    start += 1
        return ans
```


