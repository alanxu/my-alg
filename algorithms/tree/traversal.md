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

    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Intuition: Both bfs and dfs can traverse the tree and remain
        # order of 'row's.
        # If track traverse node by columns as key, we can figure out 
        # col order by sorting the key of the column table.
        # Or if we know min/max col value, we don't need to sort. We know
        # col values are concecutive because each node is related to each
        # other.
        
        # Trick: List sort remain original order if key of 2 items are same
        if not root:
            return []
        
        column_table = defaultdict(list)
        min_column, max_column = math.inf, -math.inf
        def bfs(root):
            nonlocal min_column, max_column
            q = deque([(root, 0, 0)])
            while q:
                node, r, c = q.popleft()
                column_table[c].append(node.val)
                min_column = min(min_column, c)
                max_column = max(max_column, c)
                if node.left:
                    q.append((node.left, r + 1, c - 1))
                if node.right:
                    q.append((node.right, r + 1, c + 1))
        
            
        bfs(root)
        ans = []
        for col in range(min_column, max_column + 1):
            ans.append(column_table[col])
        return ans
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

    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        column_table = defaultdict(list)
        min_column, max_column = math.inf, -math.inf
        def bfs(root):
            nonlocal min_column, max_column
            q = deque([(root, 0, 0)])
            while q:
                node, r, c = q.popleft()
                column_table[c].append((r, node.val))
                min_column = min(min_column, c)
                max_column = max(max_column, c)
                if node.left:
                    q.append((node.left, r + 1, c - 1))
                if node.right:
                    q.append((node.right, r + 1, c + 1))
        
            
        bfs(root)
        ans = []
        for col in range(min_column, max_column + 1):
            ans.append([x[1] for x in sorted(column_table[col])])
        return ans
```

### [545. Boundary of Binary Tree](https://leetcode.com/problems/boundary-of-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Fav
class Solution(object):
    def boundaryOfBinaryTree(self, root):
        # https://youtu.be/F76LIKzluKE
        ans = [root.val]
        def dfs_leftmost(node):
            if not node or not node.left and not node.right:
                return
            # Trick: Pre order print path in order
            ans.append(node.val)
            if node.left:
                dfs_leftmost(node.left)
            elif node.right:
                dfs_leftmost(node.right)
        
        def dfs_leaves(node):
            if not node:
                return

            dfs_leaves(node.left)
            
            # Exclude root for edge case [1]
            # Pre/In/Post orders are all ok, as long as left is
            # processed before right
            if node != root and not node.left and not node.right:
                ans.append(node.val)
            
            dfs_leaves(node.right)

        def dfs_rightmost(node):
            if not node or not node.left and not node.right:
                return
            if node.right:
                dfs_rightmost(node.right)
            elif node.left:
                dfs_rightmost(node.left)
            # Trick: Post order print path in reversed order
            ans.append(node.val)
        
        if not root:
            return []
        
        # Trick: Exlcude root because it is always first and
        # avoid duplicates
        dfs_leftmost(root.left)
        dfs_leaves(root)
        dfs_rightmost(root.right)
        
        return ans
```


### [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)
