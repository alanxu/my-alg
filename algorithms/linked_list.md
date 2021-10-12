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

### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        elif not l2:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

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

## Reverse

### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
    
    # Fav
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        new_head = self.reverseList(head.next)
        # Trick: Old check already hold reference to new
        # tail. 
        head.next.next = head
        # Make cur head to cur tail by clear head.next
        head.next = None
        return new_head
```

### [25. Reverse Nodes in k-Group]

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if k < 2:
            return head
        
        node = head
        for _ in range(k):
            if not node:
                return head
            node = node.next
        
        pre, cur = None, head
        for _ in range(k):
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
            
        head.next = self.reverseKGroup(cur, k)
        
        # pre is end of original current group and
        # head of new current group. cur is the head
        # of next group
        return pre
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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Fav
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return 
        
        # find the middle of linked list [Problem 876]
        # in 1->2->3->4->5->6 find 4
        # in 1->2->3->4->5 find 3
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
        # The 3->4 in first list will retained no matter odd or even
        # number of nodes
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
        # Trick: Reference of object as key of hashmap
        if head in self.visited:
            return self.visited[head]
        _node = Node(head.val)
        # This has to be before recursive call!
        self.visited[head] = _node
        _node.next = self.copyRandomList(head.next)
        _node.random = self.copyRandomList(head.random)
        return _node
```

### [146. LRU Cache](https://leetcode.com/problems/lru-cache/)

```python
class Node:
    def __init__(self, key=None, value=None):
        self.key = key
        self.val = value
        self.prev = None
        self.nxt = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head, self.tail = Node(), Node()
        self.head.nxt, self.tail.prev = self.tail, self.head

    def _add_item(self, node):
        node.prev = self.head
        node.nxt  = self.head.nxt
        self.head.nxt.prev = node
        self.head.nxt = node
        
    def _remove_item(self, node):
        node.prev.nxt = node.nxt
        node.nxt.prev = node.prev
        
    def _move_to_head(self, node):
        self._remove_item(node)
        self._add_item(node)
        
    def _pop_tail(self):
        node = self.tail.prev
        self._remove_item(node)
        return node

    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if node:
            self._move_to_head(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        if node:
            node.val = value
            self._move_to_head(node)
        else:
            if len(self.cache) == self.capacity:
                node = self._pop_tail()
                del self.cache[node.key]
            node = Node(key, value)
            self.cache[key] = node
            self._add_item(node)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

### [460. LFU Cache](https://leetcode.com/problems/lfu-cache/)

```python
class Node:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.prev = self.next = None
        self.freq = 1
        
class DLinkedList:
    def __init__(self):
        self._sentinel = Node()
        self._sentinel.next = self._sentinel.prev = self._sentinel
        self._size = 0
        
    def __len__(self):
        return self._size
    
    def append(self, node):
        node.next = self._sentinel.next
        node.prev = self._sentinel
        node.next.prev = node
        self._sentinel.next = node
        self._size += 1
        
    def pop(self, node=None):
        if self._size == 0:
            return
        
        if not node:
            node = self._sentinel.prev
            
        node.next.prev = node.prev
        node.prev.next = node.next
        self._size -= 1
        return node
        
class LFUCache:

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._size = 0
        self._cache = {}
        self._freq = defaultdict(DLinkedList)
        self._minfreq = 0
        
    def _update(self, node):
        freq = node.freq
        self._freq[freq].pop(node)
        if self._minfreq == freq and not self._freq[freq]:
            self._minfreq += 1
        
        node.freq += 1
        freq = node.freq
        self._freq[freq].append(node)

    def get(self, key: int) -> int:
        if key not in self._cache:
            return -1
        node = self._cache[key]

        self._update(node)
        return node.val
        
    def put(self, key: int, value: int) -> None:
        if self._capacity == 0:
            return
        if key in self._cache:
            node = self._cache[key]
            node.val = value
            self._update(node)
        else:
            if self._size == self._capacity:
                node = self._freq[self._minfreq].pop()
                del self._cache[node.key]
                self._size -= 1
                
            node = Node(key, value)
            self._cache[key] = node
            self._freq[1].append(node)
            self._minfreq = 1
            self._size += 1


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```


### [445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        n1, n2 = 0, 0
        cur1, cur2 = l1, l2
        
        while cur1:
            n1 += 1
            cur1 = cur1.next
        while cur2:
            n2 += 1
            cur2 = cur2.next
            
        cur1, cur2 = l1, l2
        head = None
        while n1 > 0 and n2 > 0:
            val = 0
            if n1 >= n2:
                val += cur1.val
                cur1 = cur1.next
                n1 -= 1
            if n1 < n2:
                val += cur2.val
                cur2 = cur2.next
                n2 -= 1
                
            cur = ListNode(val)
            cur.next = head
            head = cur
            
        cur, head = head, None
        carry = 0
        while cur:
            val = (cur.val + carry) % 10
            carry = (cur.val + carry) // 10 
            
            node = ListNode(val)
            node.next = head
            head = node
            
            cur = cur.next
            
        if carry:
            node = ListNode(carry)
            node.next = head
            head = node
            
        return head
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        n1 = n2 = 0
        
        head = l1
        while head:
            n1 += 1
            head = head.next
        
        head = l2
        while head:
            n2 += 1
            head = head.next
            
        head = None
        cur1, cur2 = l1, l2
        while n1 or n2:
            if n1 > n2:
                node = ListNode(cur1.val)
                node.next = head
                head = node
                n1 -= 1
                cur1 = cur1.next
            elif n1 < n2:
                node = ListNode(cur2.val)
                node.next = head
                head = node
                n2 -= 1
                cur2 = cur2.next
            else:
                node = ListNode(cur1.val + cur2.val)
                node.next = head
                head = node
                n1 -= 1
                n2 -= 1
                cur1 = cur1.next
                cur2 = cur2.next
        
        head, cur = None, head
        carry = 0
        while cur:
            v = cur.val + carry
            val = v % 10
            carry = v // 10
            cur.val = val
            
            tmp = cur.next
            cur.next = head
            head = cur
            cur = tmp
            
        if carry:
            node = ListNode(carry)
            node.next = head
            head = node
        
        return head
```

### [1650. Lowest Common Ancestor of a Binary Tree III](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        p1, p2 = p, q
        while p1 != p2:
            p1 = p1.parent if p1.parent else q
            p2 = p2.parent if p2.parent else p
        return p1
    
    
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        p_parents = set()
        while p:
            p_parents.add(p.val)
            p = p.parent
        
        while q:
            if q.val in p_parents:
                return q
            q = q.parent
```

### [708. Insert into a Sorted Circular Linked List](https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        if not head:
            head = Node(insertVal)
            head.next = head
            return head
        
        prev, cur = head, head.next
        
        while True:
            if prev.val <= insertVal <= cur.val or \
                prev.val > cur.val and (insertVal > prev.val or insertVal < cur.val):
                    prev.next = Node(insertVal, cur)
                    return head
            
            prev, cur = cur, cur.next
            if prev == head:
                break
                
        prev.next = Node(insertVal, cur)
        return head
```