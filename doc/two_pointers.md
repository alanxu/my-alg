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

```python
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