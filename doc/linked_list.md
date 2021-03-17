# Linked List
Dummy Node
Flow Fast Pointers

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