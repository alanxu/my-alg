# Greedy

Consturct the solution step by step, for each step, follow the local optimal
rule

Greedy problems usually look like "Find minimum number of something to do something" or "Find maximum number of something to fit in some conditions", and typically propose an unsorted input.

The idea of greedy algorithm is to pick the locally optimal move at each step, that will lead to the globally optimal solution.

The standard solution has \mathcal{O}(N \log N)O(NlogN) time complexity and consists of two parts:

Figure out how to sort the input data (\mathcal{O}(N \log N)O(NlogN) time). That could be done directly by a sorting or indirectly by a heap usage. Typically sort is better than the heap usage because of gain in space.

Parse the sorted input to have a solution (\mathcal{O}(N)O(N) time).

Please notice that in case of well-sorted input one doesn't need the first part and the greedy solution could have \mathcal{O}(N)O(N) time complexity, here is an example.

How to prove that your greedy algorithm provides globally optimal solution?

Usually you could use the proof by contradiction.

## Problems

### [316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # The point is to have smaller occur as early as possible
        
        if not s:
            return ''
        
        c = Counter(s)
        
        # Start of the solution: the smallest char right of which includes
        # all distinguish chars in s
        start = 0
        
        # Interate all index can be start, and pick the smallest/leftmost one
        for i in range(0, len(s)):
            if s[i] < s[start]:
                start = i
            
            # Once visited current i, check if can continue to next i
            c[s[i]] -= 1
            if c[s[i]] == 0:
                break
                
        # Clean the solution using the selected start, then append next start
        # - Remove all prefix
        # - Remove same char as s[start] in suffix
        return s[start] + self.removeDuplicateLetters(s[start:].replace(s[start], '')) 
```

### [1253. Reconstruct a 2-Row Binary Matrix](https://leetcode.com/problems/reconstruct-a-2-row-binary-matrix/)

```python
class Solution:
    def reconstructMatrix(self, upper: int, lower: int, colsum: List[int]) -> List[List[int]]:
        
        matrix = [[0] * len(colsum) for _ in range(2)]
        
        for i, s in enumerate(colsum):
            if s == 2:
                matrix[0][i] = matrix[1][i] = 1
                upper, lower = upper - 1, lower - 1
            elif s == 1:
                # Trick: First fill the difference, then distribute evenly
                if upper >= lower:
                    matrix[0][i] = 1
                    upper -= 1
                else:
                    matrix[1][i] = 1
                    lower -= 1
                    
        return matrix if upper == 0 and lower == 0 else []
```

### [1353. Maximum Number of Events That Can Be Attended](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/)

```python
from heapq import *

class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        # Trick: Sort by start date
        events = sorted(events)
        
        day = 0
        ongoing_events = []
        next_event = 0
        ans = 0
        while day <= 10 ** 5:
            # If all events has been processed
            if next_event < len(events):
                
                # If there is no events for today to attend, fast forward to
                # the next day having events
                if events[next_event][0] > day and not ongoing_events:
                    day = events[next_event][0]

                # Push all avaialbe events to ongoing list for cur day
                while next_event < len(events):
                    event = events[next_event]
                    if event[0] <= day:
                        heappush(ongoing_events, event[1])
                        next_event += 1
                    else:
                        break
            elif not ongoing_events:
                break

            # Attend meeting for the day
            while ongoing_events:
                if heappop(ongoing_events) >= day:
                    ans += 1
                    break

            day += 1
            
        return ans

```

### [435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

```python
class Solution:
    def eraseOverlapIntervals1(self, intervals: List[List[int]]) -> int:
        # Trick: Why sort?
        # [[1,100],[11,22],[1,11],[2,12]]
        # Sort by either start or end can squeeze non duplicated interverls
        # out of duplicated intervals, thus make duplicated interverls adjacent.
        # If non dup in the middle, dup will not be processed
        intervals = sorted(intervals)
        its = intervals
        
        pre = 0
        ans = 0
        for cur in range(1, len(intervals)):
            if its[cur][0] < its[pre][1]:
                if its[cur][1] < its[pre][1]:
                    pre = cur
                ans += 1
            else:
                pre = cur
        return ans
    
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals, key=lambda x: x[1])
        its = intervals
        
        pre = 0
        ans = 0
        for cur in range(1, len(intervals)):
            if its[cur][0] < its[pre][1]:
                # No need to check end, cuz th end order is gauranteed
                # always remove current even if pre.start < cur.start
                # we believe pre is optimal and just focus right side
                ans += 1
            else:
                pre = cur
        return ans
```

### [738. Monotone Increasing Digits](https://leetcode.com/problems/monotone-increasing-digits/)

```python
class Solution:
    def monotoneIncreasingDigits(self, N):
        nums = list(map(int, str(N)))
        L = len(nums)
        
        # Trick: Move pointer back and forth using while and i +/- 1
        
        # Find first cliff as i
        i = 1
        while i < L and nums[i - 1] <= nums[i]:
            i += 1
            
        # Move i backward 1 step, that is the last bigiest num
        # before first cliff
        i -= 1
        
        # Find the position of the first occurance of biggest num
        while 0 < i < L and nums[i - 1] == nums[i]:
            i -= 1
        
        print(i)
        
        return N - (N % (10 ** (L - i - 1))) - 1 if i < L - 1 else N
```

### [1057. Campus Bikes](https://leetcode.com/problems/campus-bikes/)

```python
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        import heapq
        Lw, Lb = len(workers), len(bikes)
        
        distances = []
        
        for w in range(Lw):
            distances.append([])
            for b in range(Lb):
                distance = abs(workers[w][0] - bikes[b][0]) + abs(workers[w][1] - bikes[b][1])
                distances[-1].append((distance, w, b))
            # Trick: Sort decending, and use [].pop, cus pop() is mush faster than pop(0)
            distances[-1].sort(reverse=True)
        
        # Trick: Partion items use worker or bike, and compare only the smallest in heap,
        #        in addition use the other demention with used set
        ans, used_bike = [None] * Lw, set()
        heap = [distances[i].pop() for i in range(Lw)]
        heapify(heap)
        while len(used_bike) < len(ans):
            d, w, b = heapq.heappop(heap)
            if b not in used_bike:
                ans[w] = b
                used_bike.add(b)
            else:
                heapq.heappush(heap, distances[w].pop())
                
        return ans
```

### [1029. Two City Scheduling](https://leetcode.com/problems/two-city-scheduling/)

```python
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        costs.sort(key=lambda x: x[0] - x[1])
        n = len(costs) // 2
        total = 0
        for i in range(n):
            total += costs[i][0] + costs[i + n][1]
        return total
```

### [1578. Minimum Deletion Cost to Avoid Repeating Letters](https://leetcode.com/problems/minimum-deletion-cost-to-avoid-repeating-letters/)

```python
class Solution:
    def minCost(self, s: str, cost: List[int]) -> int:
        left, ans = 0, 0
        for right in range(1, len(s)):
            if s[left] == s[right]:
                if cost[left] < cost[right]:
                    ans += cost[left]
                else:
                    ans += cost[right]
                    # If right is cheaper, delete right and kept left 
                    # position for next step calculation
                    continue 

            left = right
        return ans
```

### [452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

```python
class Solution:
    def findMinArrowShots1(self, points: List[List[int]]) -> int:
        if not points: return 0
        # Trick: Sort by end?
        points = sorted(points, key=lambda x: x[1])
        
        first_end = points[0][1]
        ans = 1
        for start, end in points:
            if start > first_end:
                ans += 1
                first_end = end
        return ans
    
    def findMinArrowShots(self, points: List[List[int]]) -> int:        
        if not points: return 0
        points = sorted(points, key=lambda x: x[1])
        
        first_end = points[0][1]
        ans = 1
        for start, end in points:
            if start > first_end:
                ans += 1
                first_end = end
            else:
                first_end = min(first_end, end)
        return ans      
                
```

### [1167. Minimum Cost to Connect Sticks](https://leetcode.com/problems/minimum-cost-to-connect-sticks/)

```python
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        # Everytime pick smallest two stick, two smallest comb become non-smallest
        import heapq
        heapq.heapify(sticks)
        cost = 0
        while len(sticks) > 1:
            stk1, stk2 = heapq.heappop(sticks), heapq.heappop(sticks)
            l = stk1 + stk2
            cost += l
            heapq.heappush(sticks, l)

        return cost
```

### [406. Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # Sort be height decending, so all allocated people is higer than curent one
        # then you can use insert()
        # Trick: Sort, be careful for second demension
        people.sort(key = lambda x: (-x[0], x[1]))
        output = []
        for p in people:
            # Trick: insert() create elements
            output.insert(p[1], p)
        return output
```



### [853. Car Fleet](https://leetcode.com/problems/car-fleet/)

```python
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        
        # Create (position, speed) array for each car to apply the formula
        cars = sorted(zip(position, speed), key=lambda x: (-x[0], x[0]))
        # Calculate the theritical time each car should take to arrive target
        times = [(target - pos) / spd for pos, spd in cars]
        # If following car take more time, we got 1 more fleet,
        # then cars following that car will take that as lead
        ans = 1 if times else 0
        lead = 0
        for i in range(len(times)):
            if times[lead] < times[i]:
                ans += 1
                lead = i
        return ans
                
    def carFleet2(self, target: int, position: List[int], speed: List[int]) -> int:
        cars = sorted(zip(position, speed))
        times = [float(target - p) / s for p, s in cars]
        print(times)
        ans = 0
        while len(times) > 1:
            lead = times.pop()
            if lead < times[-1]: ans += 1  # if lead arrives sooner, it can't be caught
            else: times[-1] = lead # else, fleet arrives at later time 'lead'

        return ans + bool(times) # remaining car is fleet (if it exists)
```

## Internal Intersection

### [1229. Meeting Scheduler](https://leetcode.com/problems/meeting-scheduler/):

```python
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        # Don't forget sort
        slots1.sort()
        slots2.sort()
        
        # Rolling two slots
        i, j = 0, 0
        while i < len(slots1) and j < len(slots2):
            # If current slot1 is completely behide slot2, move to next slot2
            if slots1[i][0] > slots2[j][1]:
                j += 1
            # If current slot2 is completely behide slot1, move to next slot1
            elif slots2[j][0] > slots1[i][1]:
                i += 1
            # If there are overlapping of current slots
            else:
                start = max(slots1[i][0], slots2[j][0])
                slot = min(slots1[i][1], slots2[j][1]) - start
                if slot >= duration:
                    return [start, start + duration]
                # Roll slot which ends earlier
                if slots1[i][1] > slots2[j][1]:
                    j += 1
                else:
                    i += 1
```

### [986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        i, j = 0, 0
        # Trick: Intervel List Intersections
        # Two pointers for each list
        first, second = firstList, secondList
        ans = []
        while i < len(firstList) and j < len(secondList):
            if first[i][0] > second[j][1]:
                # If cur first is later than cur second, fast forward second
                j += 1
            elif second[j][0] > first[i][1]:
                # If cur second is later than cur first, fast forward second
                i += 1
            else:
                # If there is intersection, get it using max/min
                ans.append([max(first[i][0], second[j][0]), min(first[i][1], second[j][1])])
                # Move the one ends ealier forward, there might be intersection with the other one
                if first[i][1] < second[j][1]:
                    i += 1
                else:
                    j += 1
        return ans
```

### [846. Hand of Straights](https://leetcode.com/problems/hand-of-straights/)

```python
class Solution:
    def isNStraightHand(self, hand: List[int], W: int) -> bool:
        hand = sorted(hand, reverse=True)
        while hand:
            start = hand.pop()
            for _ in range(W - 1):
                start += 1
                if start not in hand:
                    return False
                else:
                    hand.remove(start)
        return True
    def isNStraightHand(self, hand: List[int], W: int) -> bool:
        counts = collections.Counter(hand)
        while counts:
            # Trick: Use min() get smallest key from counter
            # Get the current smallest, there should be W consecutive
            # nums starting from it, if no, return False
            start = min(counts)
            for nxt in range(start, start + W):
                if not counts[nxt]:
                    return False
                counts[nxt] -= 1
                if counts[nxt] == 0:
                    # Have to del so min() will never return it
                    del counts[nxt]
        return True
```


### [1191. K-Concatenation Maximum Sum](https://leetcode.com/problems/k-concatenation-maximum-sum/)

```python
class Solution:
    def kConcatenationMaxSum(self, arr: List[int], k: int) -> int:
        # https://youtu.be/-T19A8DvD6U
        # When sum > 0, even if maxsum in in middle of arr, maxsum_2
        # will be acrose 2 arr...
        def kadane(arr, k):
            arr = arr * k
            global_sum = local_sum = 0
            for x in arr:
                local_sum = max(local_sum + x, x)
                global_sum = max(global_sum, local_sum)
                
            return global_sum
        
        MOD = 10 ** 9 + 7
        
        if k < 3:
            return kadane(arr, k) % MOD
        
        sum_ = sum(arr)
        ans1 = kadane(arr, 1)
        ans2 = kadane(arr, 2)
        
        return max(ans1, ans2, ans2 + sum_ * (k - 2)) % MOD
```