## BST

### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        # Intuition: BST's pre-order traverse yields
        # a increasing sequence. So we just needs
        # to pre-order traverse the tree and check
        # every node is bigger than previous one in travese.
        self.pre = -math.inf
        def dfs(node):
            if not node:
                return True
            if not dfs(node.left):
                return False
            if node.val <= self.pre:
                return False
            self.pre = node.val
            if not dfs(node.right):
                return False
            return True
        return dfs(root)
```

### [333. Largest BST Subtree](https://leetcode.com/problems/largest-bst-subtree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestBSTSubtree(self, root: TreeNode) -> int:
        self.ans = 0
        def check(node):
            # Check if the tree with node as root is a valid
            # BST, returns (num_of_nodes, max_v, min_v). If
            # the tree is not valid, return (-1, None, None).
            # If tree is a single node, (1, node.vale, node.val).
            # If tree is None, (0, None, None)
            if not node:
                return 0, None, None
            lnum, lmax, lmin = check(node.left)
            rnum, rmax, rmin = check(node.right)
            lvalid, rvalid = False, False
            curnum, curmax,curmin = -1, None, None

            if lnum >= 0:
                lvalid = lmax < node.val if lnum > 0 else True
                curmin = lmin if lnum > 0 else node.val
            if rnum >= 0:
                rvalid = rmin > node.val if rnum > 0 else True
                curmax = rmax if rnum > 0 else node.val
            
            if lvalid and rvalid:
                curnum = 1 + lnum + rnum
                self.ans = max(self.ans, curnum)
            return curnum, curmax, curmin

        check(root)
        return self.ans
```

### [285. Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        self.last = None
        def dfs(node):
            if not node:
                return
            l = dfs(node.left)
            if l:    
                return l
            if self.last and self.last == p:
                return node
            self.last = node
            r = dfs(node.right)
            if r:    
                return r
            return None
        return dfs(root)
    
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        # Time Complexity: O(N) since we might end up encountering a skewed 
        # tree and in that case, we will just be discarding one node at a time. 
        # For a balanced binary-search tree, however, the time complexity will 
        # be O(logN) which is what we usually find in practice.
        if not root: 
            return None
        if root.val > p.val:
            return self.inorderSuccessor(root.left, p) or root
        else:
            return self.inorderSuccessor(root.right, p)
```

### [510. Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/)

```python
class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Node':
        # KEY is to find the closet node bigger than node.
        # If node has right substree, the next is the smallest node on its right
        # substree, which is the very left node in substree
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        
        # If node does not have right subtree, the next is the smallest node in
        # its nearest parent that current node is in its left subtree
        while node.parent and node == node.parent.right:
            node = node.parent
        return node.parent

# Node = 13
# 15
#   6          18
#  3   7     17 20
# 2 4    13
#       9
```

### [426. Convert Binary Search Tree to Sorted Doubly Linked List](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""

class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        # Intuition: Inorder DFS
        # Use last to check if the cur node is first (left-most leaves)
        # first is the smallest. If it's first, set the first. For every
        # cur node processed in dfs, it is last, so point last to it by
        # set the left/right pointers, note only when cur node is not first.
        
        if not root:
            return root
        first = last = None
        def dfs(node):
            # Trick: DFS for tree
            # All template, logic on cur node logic
            nonlocal first, last
            if not node:
                return
            dfs(node.left)
            if not last:
                first = node
            else:
                last.right = node
                node.left = last
            last = node
            dfs(node.right)
        dfs(root)
        first.left = last
        last.right = first
        return first
```

## Unique BST

### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)

```python
class Solution:
    def numTrees(self, n: int) -> int:
        # Intuition: dp[i] is ans for num [1:i]
        # dp[i] = sum(p(j)) where p(j) is possiblities with j as
        # root () where j in [1:i].
        # p(j) = dp[left] * dp[right]
        # dp[left] is all possibilities for num [1:j-1]
        # dp[right] is all possiblilities for num [j+1, i] = [1: i-j]
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
                
        return dp[-1]
```

### [95. Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/)

```python
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        @functools.lru_cache(None)
        def dfs(start, end):
            # Thought about use start == end as termination
            # condition, to use that, you still needs to 
            # handle end > start, say for s,...,e, when root
            # is s, we need to get left, we call dfs(s, s - 1),
            # this call needs to return [None], we can handle
            # it at end but it is messy.
            if start > end:
                return [None]
            
            ans = []
            for root in range(start, end + 1):
                lefts = dfs(start, root - 1)
                rights = dfs(root + 1, end)
                
                for l in lefts:
                    for r in rights:
                        tree = TreeNode(root, l, r)
                        ans.append(tree)
            return ans
        return dfs(1, n)
```

## Others

### [449. Serialize and Deserialize BST](https://leetcode.com/problems/serialize-and-deserialize-bst/)
### [173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)