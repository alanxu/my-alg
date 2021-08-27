## Monotonic Queue

### [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

```python
class Solution:
    """
    Input: nums = [1,3,1,2,0,5], k = 3
    
    [1]
    [3]
    [3, 1]
    [3, 2]
    [2, 0]
    [5]
    """
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # Intuition: Monotonic queue and pop on both ends
        # Maintain a monotonic decreasing queue, bigger numer will pop
        # smaller ones in the queue, smaller number will append to queue.
        # The max will always in q[0]. One thing is we have to popout the
        # expired max (if q[0] is expired)
        q = deque()
        ans = []
        for i, x in enumerate(nums):
            # Pop smaller elements with the lastes x
            while q and nums[q[-1]] <= nums[i]:
                q.pop()
            
            # Expire out-of-window max if there is any
            # If there are some old value exist in q, it must be
            # the max
            while q and q[0] < i - k + 1:
                q.popleft()
                
            # Enqueue x after clean up the queue
            # Trick: Use index not value
            q.append(i)
            
            # If the window is complete, record the running max
            if i >= k - 1:
                ans.append(nums[q[0]])
        return ans
```

### [1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        
        left = 0
        max_dq, min_dq = deque(), deque()
        
        ans = 0
        
        for right, num in enumerate(nums):
            # Maintain the running max and min of the window
            # by creating two monotonic dequeue.
            # For every new number, clear all smaller/greater number in each queue.
            # New max/min number will clear all previous nums.
            # The order in queue is accending as well as by index, this is important
            # The queues are updated from right, the max/min value always at left end
            while max_dq and max_dq[-1] < num:
                max_dq.pop()
            while min_dq and min_dq[-1] > num:
                min_dq.pop()
            max_dq.append(num)
            min_dq.append(num)
            
            if max_dq[0] - min_dq[0] > limit:
                if nums[left] == max_dq[0]:
                    max_dq.popleft()
                if nums[left] == min_dq[0]:
                    min_dq.popleft()
                left += 1
            
            ans = max(ans, right - left + 1)
        
        return ans
            
        # If use heap, add (+/-value, index)
        # https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/discuss/609771/JavaC%2B%2BPython-Deques-O(N)
```

### [862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)

```python
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        """
        Intuition: Calculate prefix sums, then find two index where
        their prefix sum difference >= k.
        """
        prefixes = [0]
        for x in nums:
            prefixes.append(prefixes[-1] + x)
        
        q, ans = deque(), math.inf
        
        for i, p in enumerate(prefixes):
            while q and prefixes[i] - prefixes[q[0]] >= k:
                ans = min(ans, i - q.popleft())
            while q and prefixes[i] <= prefixes[q[-1]]:
                q.pop()
            q.append(i)
        
        return ans if ans < math.inf else -1
```

## Two Queues

### [1670. Design Front Middle Back Queue](https://leetcode.com/problems/design-front-middle-back-queue/)
```python
class FrontMiddleBackQueue:
    

    def __init__(self):
        self.A, self.B = collections.deque(), collections.deque()

    def pushFront(self, val: int) -> None:
        self.A.appendleft(val)
        self.balance()

    def pushMiddle(self, val: int) -> None:
        if len(self.A) > len(self.B):
            self.B.appendleft(self.A.pop())
        self.A.append(val)

    def pushBack(self, val: int) -> None:
        self.B.append(val)
        self.balance()

    def popFront(self) -> int:
        val = self.A.popleft() if self.A else -1
        self.balance()
        return val

    def popMiddle(self) -> int:
        val = (self.A or [-1]).pop()
        self.balance()
        return val

    def popBack(self) -> int:
        val = (self.B or self.A or [-1]).pop()
        self.balance()
        return val
    
    def balance(self) -> None:
        if len(self.A) > len(self.B) + 1:
            self.B.appendleft(self.A.pop())
        if len(self.A) < len(self.B):
            self.A.append(self.B.popleft())

# Your FrontMiddleBackQueue object will be instantiated and called as such:
# obj = FrontMiddleBackQueue()
# obj.pushFront(val)
# obj.pushMiddle(val)
# obj.pushBack(val)
# param_4 = obj.popFront()
# param_5 = obj.popMiddle()
# param_6 = obj.popBack()
```



## Others