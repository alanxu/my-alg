

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

