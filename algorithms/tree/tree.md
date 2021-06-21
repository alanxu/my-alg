

## Balanced Tree

### [729. My Calendar I](https://leetcode.com/problems/my-calendar-i/solution/)

```python
class Node:
    __slots__ = 'start', 'end', 'left', 'right'
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left = self.right = None

    def insert(self, node):
        if node.start >= self.end:
            if not self.right:
                self.right = node
                return True
            return self.right.insert(node)
        elif node.end <= self.start:
            if not self.left:
                self.left = node
                return True
            return self.left.insert(node)
        else:
            return False

class MyCalendar(object):
    def __init__(self):
        self.root = None

    def book(self, start, end):
        if self.root is None:
            self.root = Node(start, end)
            return True
        return self.root.insert(Node(start, end))
```


## Others

### [173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)
```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self.push(root)

    def next(self) -> int:
        if self.stack:
            node = self.stack.pop()
            if node.right:
                self.push(node.right)
            return node.val

    def hasNext(self) -> bool:
        if self.stack:
            return True
        return False
        
    def push(self, node):
        if node:
            self.stack.append(node)
            if node.left:
                self.push(node.left)
```

### [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        # Utilize feature of perfect binary tree that all parents have
        # both left and right.
        # Starting from root as first level, cur level is responsible
        # to handle next level, when you are in cur level, the next
        # of cur level already been set in prev level.
        # Use a lefmost pointer to mark the cur level, only point to leftmost
        # nodes, then use cur pointer to traverse cur level via next (set in 
        # prev level).
        
        leftmost = root
        # Check root is not None or not last level
        while leftmost and leftmost.left:
            cur = leftmost
            while cur:
                cur.left.next = cur.right
                cur.right.next = cur.next.left if cur.next else None
                cur = cur.next
            
            leftmost = leftmost.left
        return root
```

























### [1038. Binary Search Tree to Greater Sum Tree](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/)

```python
class Solution:
    # Inorder traverse from right, starting from max node
    # calculate accumulate val.
    val = 0
    def bstToGst(self, root: TreeNode) -> TreeNode:
        if root.right:
            self.bstToGst(root.right)
        root.val = self.val = root.val + self.val
        if root.left:
            self.bstToGst(root.left)
        return root
```




### [270. Closest Binary Search Tree Value](https://leetcode.com/problems/closest-binary-search-tree-value/)

```python
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        # Intuition: If target < root.val, it is btw min(root.left) and root.val, it must
        # not close to any(root.right), vice versa
        
        def dfs(node):
            if node.val < target and node.right:
                return min(node.val, dfs(node.right), key=lambda x: abs(target - x))
            elif node.val > target and node.left:
                return min(node.val, dfs(node.left), key=lambda x: abs(target - x))
            else:
                return node.val
            
        return dfs(root)
```





### [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        
        if not root:
            return []
        
        q = deque([root])
        
        rightside = []
        
        while q:
            length = len(q)
            
            for i in range(length):
                node = q.popleft()
                
                if i == length - 1:
                    rightside.append(node.val)
                    
                if node.left:
                    q.append(node.left)
                    
                if node.right:
                    q.append(node.right)
        return rightside
```



### [1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        nodes = []
        #node_map = 
        def traverse(node):
            if node:
                traverse(node.left)
                nodes.append(node.val)
                traverse(node.right)
            
        def construct(nodes):
            if nodes:
                l = len(nodes)
                mid = l // 2
                root = TreeNode(nodes[mid])
                root.left = construct(nodes[:mid])
                root.right = construct(nodes[mid+1:])
                return root
            
            
        traverse(root)
        return construct(nodes)
```

