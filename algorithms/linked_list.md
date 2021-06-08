# Linked List
Dummy Node
Slow Fast Pointers

## Slow Fast Pointers

### [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # Trick: Fast Slow Pointers
        slow = fast = head
        # Fast is n ahead of Flow
        for _ in range(n):
            fast = fast.next
        # If n is len(list), remove the head, so return head.next
        if not fast:
            return head.next
        # Move slow and fast together until fast points to the end
        # Trick: check fast.next not None
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        # The nth node is next to the Slow, so skip it
        # Trick: slow.next = slow.next.next
        slow.next = slow.next.next
        return head
```

### [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        intersect = self.detectIntersect(head)
        if not intersect:
            return None
        
        p1, p2 = head, intersect
        
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
            
        return p1
    
    def detectIntersect(self, head):
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return slow
        return None
```

### [148. Sort List](https://leetcode.com/problems/sort-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def merge(self, h1, h2):
        dummy = tail = ListNode(None)
        while h1 and h2:
            if h1.val < h2.val:
                tail.next, tail, h1 = h1, h1, h1.next
            else:
                tail.next, tail, h2 = h2, h2, h2.next
    
        tail.next = h1 or h2
        return dummy.next
    
    def sortList(self, head):
        if not head or not head.next:
            return head
    
        pre, slow, fast = None, head, head
        while fast and fast.next:
            pre, slow, fast = slow, slow.next, fast.next.next
        pre.next = None

        return self.merge(*map(self.sortList, (head, slow)))
```

## Recursion

### [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        
        first, second = head, head.next
        
        first.next = self.swapPairs(second.next)
        second.next = first
        
        return second
```

## Others
### [61. Rotate List](https://leetcode.com/problems/rotate-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: 'ListNode', k: 'int') -> 'ListNode':
        # base cases
        if not head:
            return None
        if not head.next:
            return head
        
        # Trick: Rotate list using ring
        # close the linked list into the ring
        old_tail = head
        n = 1
        while old_tail.next:
            old_tail = old_tail.next
            n += 1
        old_tail.next = head
        
        # find new tail : (n - k % n - 1)th node
        # and new head : (n - k % n)th node
        new_tail = head
        for i in range(n - k % n - 1):
            new_tail = new_tail.next
        new_head = new_tail.next
        
        # break the ring
        new_tail.next = None
        
        return new_head
```

### [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # Trick: Fast Slow Pointers
        slow = fast = head
        # Fast is n ahead of Flow
        for _ in range(n):
            fast = fast.next
        # If n is len(list), remove the head, so return head.next
        if not fast:
            return head.next
        # Move slow and fast together until fast points to the end
        # Trick: check fast.next not None
        while fast.next:
            fast = fast.next
            slow = slow.next
        # The nth node is next to the Slow, so skip it
        # Trick: slow.next = slow.next.next
        slow.next = slow.next.next
        return head
```

### [143. Reorder List](https://leetcode.com/problems/reorder-list/)

```python
# Fav
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return 
        
        # find the middle of linked list [Problem 876]
        # in 1->2->3->4->5->6 find 4 
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
            
        # reverse the second part of the list [Problem 206]
        # convert 1->2->3->4->5->6 into 1->2->3->4 and 6->5->4
        # reverse the second half in-place
        prev, curr = None, slow
        while curr:
            curr.next, prev, curr = prev, curr, curr.next       

        # merge two sorted linked lists [Problem 21]
        # merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
        first, second = head, prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next
```

### [82. Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # Trick: Deduplicate linked list
        dummy = ListNode(0, head)
        pre = dummy
        
        # Initially head is pre.next;
        # pre is moved from one unique value to another;
        # use head to check if there are dup ahead of pre;
        # if yes, move head to the end of dup, and set pre.next
        # to the next unique, so pre can move smoothly
        while head:
            if head.next and head.val == head.next.val:
                while head.next and head.val == head.next.val:
                    head = head.next
                pre.next = head.next
            else:
                pre = pre.next
            head = head.next
        return dummy.next
```

### [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # Trick: Swap nodes in pairs - Recursion
        if not head or not head.next:
            return head
        
        first, second = head, head.next
        
        first.next = self.swapPairs(second.next)
        second.next = first
        
        return second
```

### [61. Rotate List](https://leetcode.com/problems/rotate-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: 'ListNode', k: 'int') -> 'ListNode':
        # base cases
        if not head:
            return None
        if not head.next:
            return head
        
        # Trick: Rotate list using ring
        # close the linked list into the ring
        old_tail = head
        n = 1
        while old_tail.next:
            old_tail = old_tail.next
            n += 1
        old_tail.next = head
        
        # find new tail : (n - k % n - 1)th node
        # and new head : (n - k % n)th node
        new_tail = head
        for i in range(n - k % n - 1):
            new_tail = new_tail.next
        new_head = new_tail.next
        
        # break the ring
        new_tail.next = None
        
        return new_head
```

### [86. Partition List](https://leetcode.com/problems/partition-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        
        before = before_head = ListNode(float('-inf')) 
        after = after_head = ListNode(float('inf')) 
        
        while head:
            if head.val < x:
                before.next = head
                before = before.next
            elif head.val >= x:
                after.next = head
                after = after.next
                
            head = head.next
            
        before.next = after_head.next
        after.next = None
        
        return before_head.next
```

### [92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        dummyNode = ListNode(-1)
        dummyNode.next = head
        pre = dummyNode
        
        for _ in range(m - 1):
            pre = pre.next
            
        cur = pre.next
        reverse = None
        for _ in range(n - m + 1):
            next = cur.next
            cur.next = reverse
            reverse = cur
            cur = next
            
        pre.next.next = cur
        pre.next = reverse
        
        return dummyNode.next
```

### [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def __init__(self):
        self.visited = {}
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        
        if head in self.visited:
            return self.visited[head]
        
        n = head
        n_ = Node(n.val, None, None)
        self.visited[n] = n_
        n_.next = self.copyRandomList(n.next)
        n_.random = self.copyRandomList(n.random)
        return n_
```

### [146. LRU Cache](https://leetcode.com/problems/lru-cache/)

```python
class LinkedListNode:
    def __init__(self, key=None, val=-1):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.size = 0
        self.capacity = capacity
        self.cache = {}
        self.head, self.tail = LinkedListNode(), LinkedListNode()
        self.head.next, self.tail.prev = self.tail, self.head
    
    def _add_node(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node):
        self._remove_node(node)
        self._add_node(node)
        
    def _pop_tail(self):
        node = self.tail.prev
        self._remove_node(node)
        return node

    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if not node:
            return -1
        self._move_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        if not node:
            node = LinkedListNode(key, value)
            self.cache[key] = node
            self._add_node(node)
            self.size += 1
            if self.size > self.capacity:
                node = self._pop_tail()
                del self.cache[node.key]
                self.size -= 1
        else:
            node.val = value
            self._move_to_head(node)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```