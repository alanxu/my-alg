

## Problems

### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        class Wrapper():
            def __init__(self, node):
                self.node = node
            def __lt__(self, other):
                return self.node.val < other.node.val
        
        head = point = ListNode(0)
        q = PriorityQueue()
        for l in lists:
            if l:
                q.put(Wrapper(l))
        while not q.empty():
            node = q.get().node
            point.next = node
            point = point.next
            node = node.next
            if node:
                q.put(Wrapper(node))
        return head.next

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        heap = []
        heapq.heapify(heap)
        for i, l in enumerate(lists):
            if l:
                heapq.heappush(heap, (l.val, i))
        head = cur = ListNode()
        while heap:
            val, i = heapq.heappop(heap)
            node = lists[i]
            cur.next = node
            cur = cur.next
            node = node.next
            # Update the head for that linked list in lists, so
            # it can be found in next loop.
            lists[i] = node
            if node:
                heapq.heappush(heap, (node.val, i))
        return head.next
```

### [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)

```python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # Trick: Use Counter to create {value: count}
        count = Counter(words)
        heap = [(-freq, word) for word, freq in count.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]
```

### [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/submissions/)
```python
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        # Intuition: Heap
        # Image the surface of flood raising from 0 outside of the
        # kubes, the flood begins to go inside city when the surface
        # reached the lowest border. DFS on all nodes on the border,
        # and check their neighbours. 
        #
        # If neighbour is lower than border node, 
        # tt can hav (height - nei_height) water. Why? how about
        # the other 3 border of nei? If you are looking at the cur border
        # node, there is no other border that low than it, and if it have
        # a unvisited nei, it means the other 3 border node of that nei
        # exists, otherwise it is visted with other lower border node.
        # When the nei is filled with flood, treat is as the new border
        # same as height of cur surface, in this way we can make sure all
        # places is filled as much as possible.
        # 
        # If neighbore is higher than border node, It is above flood surface,
        # add it to heap for later when flood is higher.
        # 
        # Why we can add to a node to visited whenever we reach it? If it is lower
        # it will be filled and added to heap as a border, it is no longer required
        # to calculate the water volumn. If it is higher, it means it has lower border
        # so it can contain 0 water. Only need to visit nodes that can trap water.
        # Borders just needs to be added to heap, no needs to be visited as nei.
        #
        # In the heap, the items at front are for cur flood height, then there are
        # higher borders for next heights.
        M, N = len(heightMap), len(heightMap[0])
        heap, visited = [], set()
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        for i in range(M):
            for j in range(N):
                if i == 0 or j == 0 or i == M - 1 or j == N - 1:
                    heapq.heappush(heap, (heightMap[i][j], i, j))
                    visited.add((i, j))
        
        ans = 0
        while heap:
            height, i, j = heapq.heappop(heap)
            for d_i, d_j in dirs:
                nei_i, nei_j = i + d_i, j + d_j
                if 0 <= nei_i < M and 0 <= nei_j < N and (nei_i, nei_j) not in visited:
                    nei_height = heightMap[nei_i][nei_j]
                    ans += max(0, height - nei_height)
                    heapq.heappush(heap, (max(nei_height, height), nei_i, nei_j))
                    visited.add((nei_i, nei_j))
        
        return ans
```

### [778. Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/)

### [218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # Create events for start and end of each building.
        # The order of event properties are (left, height, right) so it can
        # sort properly based on occurance of x.
        # Start event needs both r, l, h info, cuz it control a lot of things.
        # 0 height is used to indicate an end event, since we use left as position
        # of event, we use building.right as the possition of end event of that
        # building. Right of an end event is not used
        events = []
        for b in buildings:
            l, r, h = b[0], b[1], b[2]
            # It is important to convert h to negative at this stage rather than handle it later
            # because doing it here will make sure when 2 events happen at same position,
            # the bigger height will be processed first and current max height (live[0]) is correct
            events.append((l, -h, r))
            events.append((r, 0, 0))
        events.sort()
        
        live = [(0, float('inf'))]
        skyline = []
        import heapq
        for evt in events:
            l, h, r = evt
            pos = l
            # On any events, clean all ended buildings
            # Trick: Sometimes you want to clean items in a heap/queue that not
            # easy to locat, you can attache some info to the item, and clean it
            # when it become active (atop of heap/queue)
            while live and live[0][1] <= pos: heapq.heappop(live)
            
            # On start events, push the building into live heap.
            # The live building needs info: height and right;
            # Live heap sorted by height, so we know the highest live building;
            # The right info is important, cuz it's needed to removed live building
            if h < 0:
                heapq.heappush(live, (h, r))
                
            # If the current max height (live[0]) is changed from previous skyline point (skyline[-1]),
            # The current max height is obtained by cleaning ended building and adding active building
            # in previous 2 steps;
            # The new skyline point is new max height at the current event position, the position
            # and the event doesn't necessaryly related to same building
            # We don't have edge case when no active building in live
            # because we have a (0, inf) in live which is ground that make sense.
            if not skyline or skyline[-1][1] != -live[0][0]:
                skyline.append((pos, -live[0][0]))
                
        return skyline
```

### [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

```python
from heapq import *

class MedianFinder:
    # Alg: Two heaps
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small = []
        self.large = []

    def addNum(self, num: int) -> None:
        heappush(self.small, -heappushpop(self.large, num))
        if len(self.small) > len(self.large):
            heappush(self.large, -heappop(self.small))
        
    def findMedian(self) -> float:
        # Trick: heapq, top of heap is [0]
        if len(self.large) > len(self.small):
            return self.large[0]
        return (self.large[0] - self.small[0]) / 2.0
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        heap = [(-freq, cnt) for cnt, freq in count.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]
```

### [373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # Intuition: The sum of 2 first element is guaranteed to be smallest, use a heap
        # to come up with results. For each valid pair (i, j), the next pair could be
        # either (i, j + 1) or (i + 1, j). So once pop out one pair, add max 2 next pairs
        # into the queue. 
        # A visited set has to be used to avoid duplicated pairs to be added. Why there is
        # duplication? Cuz when (i, j) is popped up, there is possibility for (i - 1, j + 1)
        # or (i + 1, j - 1) is added with (i, j) same time and it is possible that one of them
        # are poped first, thus (i, j + 1) or (i + 1, j) could be added as the neibor of one of
        # the 2.
        # 1 3 4 7
        # 1 5 17 18
        M, N = len(nums1), len(nums2)
        heap = [(nums1[0] + nums2[0], 0, 0)]
        ans, visited = [], {(0, 0)}
        while heap and len(ans) < k:
            s, i, j = heapq.heappop(heap)
            ans.append((nums1[i],nums2[j]))
            if i < M - 1 and (i + 1, j) not in visited:
                heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
                visited.add((i + 1, j))
            if j < N - 1 and (i, j + 1) not in visited:
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
                visited.add((i, j + 1))
        return ans
```


### [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for x, y in points:
            heapq.heappush(heap, (-(x * x + y * y), x, y))
            if len(heap) > k:
                heapq.heappop(heap)
        return [[x, y ]for (dst, x, y) in heap]
```