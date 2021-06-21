## Traversal of Tree
### [103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q, ans = deque([None, root]), []
        order = 1
        level_list = deque()
        while q:
            node = q.pop()
            if node:
                # If node not None, meaning cur level not complete
                
                # Construct cur level_list as per order
                if order:
                    level_list.append(node.val)
                else:
                    level_list.appendleft(node.val)
                
                # Enqueue next level nodes
                if node.left:
                    q.appendleft(node.left)
                if node.right:
                    q.appendleft(node.right)
            else:
                # If node is None, means cur level is complete and next
                # level is 100% in queue
                
                # Collect ans for cur level
                ans.append(level_list)
                # Clean level_list for next level
                level_list = deque()
                # Reverse the order
                order ^= 1
                # Add only if there are next level
                if q:
                    q.appendleft(None)
        return ans
```

### [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q = collections.deque()
        q.appendleft(root)
        ans = []
        while q:
            Len = len(q)
            level = []
            for i in range(Len):
                node = q.pop()
                level.append(node.val)
                if node.left: q.appendleft(node.left)
                if node.right: q.appendleft(node.right)
            ans.append(level)
        return ans
```

### [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        ans = []
        def dfs(node):
            ans.append(node.val)
            if node.left:
                dfs(node.left)
            if node.right:
                dfs(node.right)
        dfs(root)
        return ans
    
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        stack, ans = [root], []
        
        while stack:
            node = stack.pop()
            ans.append(node.val)
            # Trick: Use stack for dfs, add right first so left can be
            #   processed first.
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            
        return ans
```

### [314. Binary Tree Vertical Order Traversal](https://leetcode.com/problems/binary-tree-vertical-order-traversal/)

```python
from collections import defaultdict
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        columnTable = defaultdict(list)
        queue = deque([(root, 0)])

        while queue:
            node, column = queue.popleft()

            if node is not None:
                columnTable[column].append(node.val)
                
                queue.append((node.left, column - 1))
                queue.append((node.right, column + 1))
                        
        return [columnTable[x] for x in sorted(columnTable.keys())]
```



### [987. Vertical Order Traversal of a Binary Tree](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        node_list = []
        
        def bfs(root):
            q = deque([(root, 0, 0)])
            while q:
                n, r, c = q.popleft()
                
                if n:
                    node_list.append((c, r, n.val))

                    q.append((n.left, r + 1, c - 1))
                    q.append((n.right, r + 1, c + 1))
                
                
        bfs(root)
        
        node_list.sort()
        
        ret = OrderedDict()
        for column, row, value in node_list:
            if column in ret:
                ret[column].append(value)
            else:
                ret[column] = [value]

        return ret.values()
```