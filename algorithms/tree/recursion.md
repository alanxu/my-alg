## LCA

### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def find_lca(root, p, q):
            # Find lca of p or q, return None if no result find
            if not root or root == p or root == q:
                return root
            # Find lca for p or q on root's left and right sub tree
            left = find_lca(root.left, p, q)
            right = find_lca(root.right, p, q)
            
            # If left and right both non-None, p and q are on different side of root,
            # then root is ans;
            # Otherwise, both p and q must be on left or right;
            # There must be an ans.
            return root if left and right else left or right
        return find_lca(root, p, q)

    def find_one(self, node, target):
        if not node:
            return False
        if node.val == target.val:
            return True
        return self.find_one(node.left, target) or self.find_one(node.right, target)
    
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        left, right = self.lowestCommonAncestor(root.left, p, q), self.lowestCommonAncestor(root.right, p, q)
        if left or right:
            return left or right
        elif self.find_one(root, p) and self.find_one(root, q):
            return root
        else:
            return None
```

### [1650. Lowest Common Ancestor of a Binary Tree III](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/)


### [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        parent_val = root.val
        p_val = p.val
        q_val = q.val
        if p_val > parent_val and q_val > parent_val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p_val < parent_val and q_val < parent_val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
```

### [1644. Lowest Common Ancestor of a Binary Tree II](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # Intuition: Use DFS/BFS to build dict of parents of each nodes
        # then use linear search to find all ancestor of p, 
        # then use linear search to find the ancestor of q which in p.
        
        parents = {root.val:None}
        stack = [root]
        while stack:
            node = stack.pop(0)
            if node.left:
                parents[node.left.val] = node
                stack.append(node.left)
            if node.right:
                parents[node.right.val] = node
                stack.append(node.right)
        ancestors = set()
        if p.val not in parents or q.val not in parents: return None
        while p:
            ancestors.add(p.val)
            p = parents[p.val]
        while q and q.val not in ancestors:
            q = parents[q.val]
        return q
```



## Recursion



### [114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        last = None
        def preorder(node):
            if not node: return
            nonlocal last
            if last: last.right = node
            last = node
            left, right, node.left, node.right = node.left, node.right, None, None
            preorder(left)
            preorder(right)
            
        preorder(root)

    def flatten(self, root: TreeNode) -> None:
        self.previous_right = None
        def helper(root = root):
            if root:
                helper(root.right)
                helper(root.left)
                root.right, self.previous_right = self.previous_right, root
                root.left = None
        helper()

    def flatten(self, root: TreeNode) -> None:
        last = TreeNode()
        def dfs(node):
            if not node:
                return
            nonlocal last
            left, right = node.left, node.right
            node.left, node.right = None, None
            last.right = node
            last = node
            dfs(left)
            dfs(right)
        dfs(root)
```

### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_path_sum = float('-inf')
        def max_one_child(node):
            """
            Use DFS to recursively calculate max one_child_sum of each node.
            With this info in hand, a node can easily calculate the path sum
            with itself as the root (lowest node).
            So the DFS is finding max one-child sum, but in mean while we can keep 
            calculate path sum for each node as root and track the max.
            """
            if not node:
                return 0
            max_left = max_one_child(node.left)
            max_right = max_one_child(node.right)
            # Trick: Put 0 in max compasion, if both child path < 0, the max path for
            # root is just itself
            # Trick: Add one child to analyze path sum
            max_root = node.val + max(0, max_left, max_right)
            max_path_sum = node.val + max(0, max_left) + max(0, max_right)
            
            self.max_path_sum = max(self.max_path_sum, max_path_sum)
            
            return max_root
        
        max_one_child(root)
        return self.max_path_sum
```

### [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def depth(node):
            if not node:
                return 0
            nonlocal ans
            depth_left, depth_right = depth(node.left), depth(node.right)
            ans = max(ans, depth_left + 1 + depth_right)
            return 1 + max(depth_left, depth_right)
        depth(root)
        return ans - 1
```


### [1026. Maximum Difference Between Node and Ancestor](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/)

```python
# Fav
class Solution:
    def maxAncestorDiff(self, root: TreeNode) -> int:
        def dfs(node, cur_max, cur_min):
            # The distance only calculate at leaf level
            if not node:
                return cur_max - cur_min
            
            # For each node, calculate max and min and pass on
            # to next level.
            # Before and after recursive call matters, the return
            # value does not impact any behavior
            cur_max = max(cur_max, node.val)
            cur_min = min(cur_min, node.val)
            left = dfs(node.left, cur_max, cur_min)
            right = dfs(node.right, cur_max, cur_min)
            return max(left, right)
        
        return dfs(root, -math.inf, math.inf)
    def maxAncestorDiff(self, root: TreeNode) -> int:
        # Intuition: dfs(node) returns max and min of tree with node as root.
        # While calculate max/min, update ans with max distance between node
        # and max/min of it's subtrees.
        ans = -1
        def dfs(node):
            nonlocal ans
            max_v = min_v = node.val
            if node.left:
                left_max, left_min = dfs(node.left)
                ans = max(ans, abs(node.val - left_max), abs(node.val - left_min))
                max_v, min_v = max(max_v, left_max), min(min_v, left_min)
            if node.right:
                right_max, right_min = dfs(node.right)
                ans = max(ans, abs(node.val - right_max), abs(node.val - right_min))
                max_v, min_v = max(max_v, right_max), min(min_v, right_min)
            return max_v, min_v
        dfs(root)
        return ans
```



### [979. Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def distributeCoins(self, root: TreeNode) -> int:
        # Intuition: The ans is sum of moves on each node
        # with its children. 
        # Why children, it can go to parent? Because if
        # you count moves to parent, in a recursive approach,
        # you get duplicates.
        # dfs(node) returns the gap of coins when perform
        # the question on subtree with root node. With returned
        # value, we can know how many moves need to happen b/w
        # node and its parent (to/from)
    
        self.ans = 0
        def dfs(node):
            if not node:
                return 0
            # Get how many coins more/less to fullfill the question
            left, right = dfs(node.left), dfs(node.right)
            
            # The parent (node) needs to meet the requirement, take/give
            # coins as required by children, the move happens ONLY on node
            # b/w node and its children is abs(left) + abs(right).
            # Keep sum of moves on all nodes b/w their children as the ans.
            # Use abs() because we care num of moves from/to node.
            self.ans += abs(left) + abs(right)
            
            # The delta of coins of subtree with root node is how many coins
            # remaining or required, remember leave 1 for node (parent) by -1.
            # This number will be sum to ans in upper recursion.
            return node.val + left + right - 1
        dfs(root)
        return self.ans
```

### [968. Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        self.ans = 0
        covered = {None}
        
        def dfs(node=root, par=None):
            if node:
                dfs(node.left, node)
                dfs(node.right, node)
                # Bottom up, check after recursive call. Leaves get processed first.
                # If cur node is root, or left/right not covered, must mark cur node
                # as camera, cuz it is bottom-up this is last chance to get left/right
                # covered.
                # If the node is uncovered, it will not immediately be marked, cuz it is not
                # root and there is chance to cover it later on upper level.
                # This is a greedy approach, cuz second last level always marked. Whether mark
                # the root depends on structure of tree.
                #
                # Why bottom up? Because in Binary Tree path, #children >= #parent, so from
                # bottom up, we have opptunity to put camera on parent which is more optimal
                
                if not par and node not in covered or node.left not in covered or node.right not in covered:
                    self.ans += 1
                    covered.update({par, node, node.left, node.right})
                    
        dfs()
        return self.ans
```

### [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)

```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        ans = []
        # Trick: Use remining sum for path sum
        def dfs(node=root, path=[], remaining_sum=targetSum):
            if not node:
                return
            
            path.append(node.val)
            
            if node.val == remaining_sum and not node.left and not node.right:
                ans.append(path[:])
            else:
                dfs(node.left, path, remaining_sum - node.val)
                dfs(node.right, path, remaining_sum - node.val)
            
            path.pop()
            
        dfs()
        return ans
```

### [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        # Prefix sum is widely used to resolve subset sum problems.
        # In this problem, the challenges are
        # - Maintain prefix sum for the current path: Use preorder and
        #   reverse the change to the cursum once left/right is processed
        # - How to track pre sum of indirect prevous nodes in one path: You
        #   dont have to, instead you can just use dict to save the presum of 
        #   previous node and the counts of same prefix sum before
        
        pre_sums = defaultdict(int, {0: 1})
        ans = 0
        def dfs(node, cursum=0):
            if not node:
                return
            # Trick: use nonlocal
            nonlocal ans
            cursum += node.val
            ans += pre_sums[cursum - sum]
            pre_sums[cursum] += 1
            
            dfs(node.left, cursum)
            dfs(node.right, cursum)
            
            # Must only maintain the presum for current path,
            # remove the current 
            pre_sums[cursum] -= 1
        
        dfs(root)
        return ans

    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        self.ans = 0
        def dfs(node=root, target=targetSum):
            # Intuition: 2-level Revursion
            # DFS over the tree, for each node, check how
            # many pathes STARTed with cur node
            if not node:
                return
            check(node, target)
            dfs(node.left, target)
            dfs(node.right, target)
            
        def check(node, target):
            # Check starting from cur node, how may matching path
            # found
            if not node:
                return
            if node.val == target:
                self.ans += 1
            check(node.left, target - node.val)
            check(node.right, target - node.val)
            
        dfs()
        return self.ans
```

### [222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        # Intuition
        # If left sub tree height equals right sub tree height then,
        #   a. left sub tree is perfect binary tree
        #   b. right sub tree is complete binary tree
        # If left sub tree height greater than right sub tree height then,
        #   a. left sub tree is complete binary tree
        #   b. right sub tree is perfect binary tree
        if not root:
            return 0
        left_depth = self.get_depth(root.left)
        right_depth = self.get_depth(root.right)
        # Trick: Number of nodes in a complete binary tree:
        # pow(2, depth_of_the tree) - 1
        # In below, plus 1 to cound root 
        if left_depth == right_depth:
            return 1 + pow(2, left_depth) - 1 + self.countNodes(root.right)
        elif left_depth > right_depth:
            return 1 + self.countNodes(root.left) + pow(2, right_depth) - 1
        
    def get_depth(self, node):
        if not node:
            return 0
        return 1 + self.get_depth(node.left)
    
    def countNodes(self, root: TreeNode) -> int:
        def compute_depth(node):
            """
            Trick: Compute tree's depth
            A node’s height is the number of edges to its most distant leaf node. On the other hand, a node’s depth 
            is the number of edges back up to the root. So, the root always has a depth of 0 while leaf nodes always 
            have a height of 0. And if we look at the tree as a whole, its depth and height are both the root height.
            """
            d = 0
            while node.left:
                node = node.left
                d += 1
            return d
        
        def leaf_exists(idx, node, depth):
            # Trick: Binery search to check if leaf exists
            left, right = 0, 2**d - 1
            for _ in range(depth):
                pivot = left + (right - left) // 2
                if idx <= pivot:
                    node = node.left
                    right = pivot
                else:
                    node = node.right
                    left = pivot + 1
            return node is not None
        
        if not root:
            return 0
        
        d = compute_depth(root)
        print(d)
        if d == 0:
            return 1
        
        left, right = 0, 2**d - 1
        while left <= right:
            pivot = left + (right - left) // 2
            if leaf_exists(pivot, root, d):
                left = pivot + 1
            else:
                right = pivot - 1
        
        # Trick: All nodes above last level in a complete binary tree: 2^0 + 2^1 + ... + 2^(d-1) = 2^d - 1
        # Trick: All nodes in complete binary tree: 2^d - 1 + (number of leaves)
        return 2**d - 1 + left

    def countNodes(self, root: TreeNode) -> int:
        return 1 + self.countNodes(root.right) + self.countNodes(root.left) if root else 0
```

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
                uid = trees[node.val, lookup(node.left), lookup(node.right)]
                count[uid] += 1
                if count[uid] == 2:
                    ans.append(node)
                return uid
        lookup(root)
        return ans
```

```python
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        counts = defaultdict(int)
        ans = set()
        def dfs(node=root):
            if not node:
                return [None]
            struct = [node.val] + dfs(node.left) + dfs(node.right)
            counts[tuple(struct)] += 1
            if counts[tuple(struct)] == 2:
                ans.add(node)
            return struct
        dfs()
        return ans
```

### [337. House Robber III](https://leetcode.com/problems/house-robber-iii/)

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        def dfs(node=root):
            if not node:
                return (0, 0)
            left = dfs(node.left)
            right = dfs(node.right)
            
            rob = node.val + left[1] + right[1]
            not_rob = max(left) + max(right)
            
            return [rob, not_rob]
        return max(dfs())
```

### [100. Same Tree](https://leetcode.com/problems/same-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        def dfs(a, b):
            if a is None or b is None:
                return a is b
            if a.val != b.val:
                return False
            return dfs(a.left, b.left) and dfs(a.right, b.right)
        return dfs(p, q)
```

### [1110. Delete Nodes And Return Forest](https://leetcode.com/problems/delete-nodes-and-return-forest/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        to_delete = set(to_delete)
        self.ans = [root] if root.val not in to_delete else []
        def dfs(node):
            if not node:
                return
            left, right = node.left, node.right
            if node.val in to_delete:
                if left and left.val not in to_delete:
                    self.ans.append(left)
                if right and right.val not in to_delete:
                    self.ans.append(right)
                    
            if left and left.val in to_delete:
                node.left = None
            if right and right.val in to_delete:
                node.right = None
            
            dfs(left)
            dfs(right)
        dfs(root)
        return self.ans
    
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        ans = []
        def dfs(node=root, parent=None, side='left',):
            if node:
                if node.val in to_delete:
                    if node.left and node.left.val not in to_delete: 
                        ans.append(node.left)
                    if node.right and node.right.val not in to_delete: 
                        ans.append(node.right)
                    if parent:
                        if side == 'left':
                            parent.left = None
                        elif side == 'right':
                            parent.right = None
                elif node == root:
                    ans.append(node)

                dfs(node.left, node, 'left')
                dfs(node.right, node, 'right')
                
                
        dfs(root)
        return ans
```



### [814. Binary Tree Pruning](https://leetcode.com/problems/binary-tree-pruning/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if root.val == 0 and not root.left and not root.right:
            return None
        return root
```


### [938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        ans = 0
        def dfs(node):
            nonlocal ans
            if not node:
                return
            if low <= node.val <= high:
                ans += node.val
            if node.val > low:
                dfs(node.left)
            if node.val < high:
                dfs(node.right)
        dfs(root)
        return ans
```


### [1120. Maximum Average Subtree](https://leetcode.com/problems/maximum-average-subtree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        self.ans = -math.inf
        def dfs(root):
            if not root:
                return 0, 0
            n, s = 1, root.val
            left_n, left_s = dfs(root.left)
            right_n, right_s = dfs(root.right)
            n = 1 + left_n + right_n
            s = root.val + left_s + right_s
            
            self.ans = max(self.ans, s / n)
            
            return n, s
        dfs(root)
        return self.ans
```


### [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```


###