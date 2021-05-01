

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