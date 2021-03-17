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

