## Find Duplicate Subtrees

### [652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/)
```python
class Solution(object):
    def findDuplicateSubtrees(self, root):
        trees = collections.defaultdict()
        # Trick: Generate unique increasing (0, 1, 2, ...) ids for unique keys
        trees.default_factory = trees.__len__
        count = collections.Counter()
        ans = []
        def lookup(node):
            if node:
                # Trick: Generate empty value for leaf's children to avoid duplicate cases
                # 572. Subtree of Another Tree
                uid = trees[node.val, lookup(node.left), lookup(node.right)]
                count[uid] += 1
                if count[uid] == 2:
                    ans.append(node)
                return uid
        lookup(root)
        return ans
```

### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def match(self, s, t):
        # Trick: If s or t is None, (s and t) returns False, if s and t both are None,
        # (s and t) also return False, so (s is t) is True if both None, False if one
        # is None.
        if not (s and t):
            return s is t
        if s.val == t.val and self.match(s.left, t.left) and self.match(s.right, t.right):
            return True
        return False
    
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        # O(M * N)
        if self.match(root, subRoot):
            return True
        if not root:
            return False
        if self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot):
            return True
        return False
```

### [1367. Linked List in Binary Tree](https://leetcode.com/problems/linked-list-in-binary-tree/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def match(self, head, root):
        if not head: return True
        elif not root: return False
        return head.val == root.val and (self.match(head.next, root.left) or self.match(head.next, root.right))
    
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        if not head: return True
        elif not root: return False
        return self.match(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)
```