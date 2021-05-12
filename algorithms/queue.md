## Monotonic Queue

### [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque
        dq = deque()
        
        def clean_queue(i):
            # If the (i - k)th element is still in queue, it is the max so far, or it has been
            # poped, new element only come from the right
            if dq and dq[0] == i - k:
                dq.popleft()
                
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()

        ans = []
        
        for i in range(len(nums)):
            # Before add each num, remove the edge element and remove all element < nums[i]
            # This will maintain a queue with decending order and keep nums[0] as biggest one
            # Index is pushed to queue
            clean_queue(i)
            
            # Append current element index
            dq.append(i)
            
            # Needs to wait for first k elements
            if i >= k - 1:
                ans.append(nums[dq[0]])
            
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

## Others