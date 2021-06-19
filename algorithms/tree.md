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

## BFS

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

### [117. Populating Next Right Pointers in Each Node II](117. Populating Next Right Pointers in Each Node II)
```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root: return root
        q = collections.deque()
        q.appendleft(root)
            
        while q:
            Len = len(q)
            for i in range(Len):
                node = q.pop()
                if node.left: q.appendleft(node.left)
                if node.right: q.appendleft(node.right)
                if i < Len - 1:
                    node.next = q[-1]
                
        return root
```

## DFS

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

## Tree to Graph

### [863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # Intuition: Tread tree as undirectional graph, then BFS for k levels.
        
        # Build graph
        graph = defaultdict(list)
        def connect(parent, child):
            if not child:
                return
            if parent:
                graph[parent.val].append(child.val)
                graph[child.val].append(parent.val)
            connect(child, child.left)
            connect(child, child.right)
        connect(None, root)
        
        # BFS starting from target, do it k times
        # nodes remaining in the q is the ans
        q, visited = deque([target.val]), {target.val}
        for _ in range(k):
            # Trick: Use len, each loop, the reaming
            # nodes in the q are ALL nodes of next level
            l = len(q)
            for _ in range(l):
                node = q.pop()
                for nei in graph[node]:
                    if nei not in visited:
                        q.appendleft(nei)
                        visited.add(nei)
        return list(q)
```

### [742. Closest Leaf in a Binary Tree](https://leetcode.com/problems/closest-leaf-in-a-binary-tree/)

```python
class Solution:
    def findClosestLeaf(self, root: TreeNode, k: int) -> int:
        graph, leaves = defaultdict(list), set()
        
        def build_graph(node):
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
                build_graph(node.left)
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)
                build_graph(node.right)
            if not node.left and not node.right:
                leaves.add(node.val)
                
        build_graph(root)
        # Trick: Use val as the keys since it is unique, save lookup time
        q = collections.deque([k])
        # Trick: Use memory to skip visited node in graph traverse
        seen = set(q)
        
        # Trick: No need to mark end of each level in this problem
        while q:
            val = q.pop()
            if val in leaves:
                return val
            for nei in graph[val]:
                if nei not in seen:
                    seen.add(nei)
                    q.appendleft(nei)
```

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

## Recursion

### [297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    # Intuition: 
    # First, the key is to nail down the serial
    # format: <root>,<left_tree>,<right_tree>.
    # This is the base of a recursion method.
    # Second: Fill all the leaves with 'null',
    # this makes it possible for deser to understand.
    # With a 'full' binary tree, the dfs even doesn't
    # need to check empty.
    
    

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return 'null,'
        # Trick: Only add "," for real step in
        # a recursion method, this needs to be
        # understood very well.
        ans = str(root.val) + ","
        ans += self.serialize(root.left)
        ans += self.serialize(root.right)
        return ans
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def rdeserialize(data):
            # No need check empty, if cur root
            # is 'null', it won't continue.
            # No need to strip endding ',', cuz
            # it will always next to a 'null',
            # so it will return.
            x = data.popleft()
            if x== 'null':
                return None

            root = TreeNode(x)
            root.left = rdeserialize(data)
            root.right = rdeserialize(data)
            return root
        return rdeserialize(deque(data.split(',')))

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
```

### [536. Construct Binary Tree from String](https://leetcode.com/problems/construct-binary-tree-from-string/)
```python
# Fav
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def str2tree(self, s: str) -> TreeNode:
        if not s:
            return None
        # Intuition: Use pointer i to mark cur pos,
        # If cur is char, append it to value of cur
        # node/root, 
        # if cur is '(', we meet cur's child, if cur
        # root is None, meaning the following chars are
        # for cur root's left child, so create the root,
        # then do a recursive call with i + 1, otherwise
        # root is not None, meaning we meet cu's right
        # child, the root has already be created before left
        # child is processed.
        # If cur is ')', the current node is done, so just
        # return it and i + 1
        L = len(s)
        def build(i=0):
            root_val, root = [], None
            while i < L:
                if s[i] in '()':
                    if not root:
                        root = TreeNode(''.join(root_val))
                    if s[i] == '(':
                        if not root.left:
                            root.left, i = build(i + 1)
                        else:
                            root.right, i = build(i + 1)
                    else:
                        return root, i + 1
                else:
                    root_val.append(s[i])
                    i += 1
            # To handle case there is no '()' behide root
            return root if root else TreeNode(''.join(root_val))
        return build()
```

### [106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```python
# Fav
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # For a given range (lower, upper) initially (0, n - 1);
        # Last ndoe in postorder is the root;
        # Given root, we can find the left/right subtree range in inorder;
        # Exlude last node in postorder, remmaining = (postorder left subtree )+ (postorder right substree)
        # Starting from right, when right subtree is done, the remaining postorder is for left subtree
        def build(lower, upper):
            if lower > upper:
                return None
            
            val = postorder.pop()
            idx = idx_map[val]
            root = TreeNode(val)
            
            # Right first
            root.right = build(idx + 1, upper)
            root.left = build(lower, idx - 1)
            
            return root

        idx_map = {v: i for i, v in enumerate(inorder)}
        
        return build(0, len(inorder) - 1)
```

### [889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

```python
class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        # Don't try to understand how each single node move.
        # KEY is for current step, get the root, then figure out the correct input for next interate
        # For each interate, the input of pre/post must be a VALID pre/post of a smaller tree
        
        # Handle edge case first
        # For each interate, get root from first pre
        if not pre: return None
        root = TreeNode(pre[0])
        if len(pre) == 1: return root
        
        # For each interate, Both pre and post can be sepeated to 3 parts:
        # pre  = root + pre left + pre right
        # post = post left + post right + root
        # We need to find each part and call next resursion
        
        # Get root of left sub tree and use it as seperator
        # L is the length of current left tree, beause it is the last element in left tree in post
        L = post.index(pre[1]) + 1
        
        # When L is figured out:
        # pre left  = pre[1:L+1], pre right  = pre[L+1:]
        # post left = post[:L],   post right = post[L:-1]
        root.left  = self.constructFromPrePost(pre[1:L+1], post[:L])
        root.right = self.constructFromPrePost(pre[L+1:], post[L:-1])
        return root
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




### [979. Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/)

```python
class Solution(object):
    def distributeCoins(self, root):
        self.ans = 0

        def dfs(node):
            if not node: return 0
            L, R = dfs(node.left), dfs(node.right)
            self.ans += abs(L) + abs(R)
            return node.val + L + R - 1

        dfs(root)
        return self.ans
```

### [968. Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/)

```python
class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        self.ans = 0
        covered = {None}
        
        def dfs(node=root, par=None):
            if node:
                dfs(node.left, node)
                dfs(node.right, node)
                # Trick: If the logic is after recursion call, it first run on bottom then up
                #        while if it is before recursion call, it first run or top then bottom
                if not par and node not in covered or node.left not in covered or node.right not in covered:
                    self.ans += 1
                    covered.update({par, node, node.left, node.right})
                    
        dfs()
        return self.ans
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

### [606. Construct String from Binary Tree](https://leetcode.com/problems/construct-string-from-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def tree2str(self, t: TreeNode) -> str:
        # Intuition: Only add quotes at parent level
        def dfs(node):
            if not node:
                return ''
            if node.right:
                return f'{node.val}({dfs(node.left)})({dfs(node.right)})'
            elif node.left:
                return f'{node.val}({dfs(node.left)})'
            else:
                return f'{node.val}'
        return dfs(t)
    
    def tree2str(self, t: TreeNode) -> str:
        if not t:
            return ''
        stack, visited, ans = [t], set(), ''
        while stack:
            node = stack[-1]
            if node in visited:
                stack.pop()
                ans += ')'
            else:
                visited.add(node)
                ans += f'({node.val}'
                if not node.left and node.right:
                    ans += '()'
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
        return ans[1:-1]
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



### [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        
        # Recursive solution
        if inorder:   
            # Find index of root node within in-order traversal
            index = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[index])
            
            # Recursively generate left subtree starting from 
            # 0th index to root index within in-order traversal
            root.left = self.buildTree(preorder, inorder[:index])
            
            # Recursively generate right subtree starting from 
            # next of root index till last index
            root.right = self.buildTree(preorder, inorder[index+1:])
            return root
```

### [106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # For a given range (lower, upper) initially (0, n - 1);
        # Last ndoe in postorder is the root;
        # Given root, we can find the left/right subtree range in inorder;
        # Exlude last node in postorder, remmaining = (postorder left subtree )+ (postorder right substree)
        # Starting from right, when right subtree is done, the remaining postorder is for left subtree
        def build(lower, upper):
            if lower > upper:
                return None
            
            val = postorder.pop()
            idx = idx_map[val]
            root = TreeNode(val)
            
            # Right first
            root.right = build(idx + 1, upper)
            root.left = build(lower, idx - 1)
            
            return root

        idx_map = {v: i for i, v in enumerate(inorder)}
        
        return build(0, len(inorder) - 1)
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

### [449. Serialize and Deserialize BST](https://leetcode.com/problems/serialize-and-deserialize-bst/)

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        """
        def postorder(node):
            return postorder(node.left) + postorder(node.right) + [node.val] if node else []
        return ','.join(map(str, postorder(root)))

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        def redeserialize(lower=-math.inf, upper=math.inf):
            if not data or data[-1] < lower or data[-1] > upper:
                return None
            val = data.pop()
            root = TreeNode(val)
            root.right = redeserialize(val, upper)
            root.left = redeserialize(lower, val)
            return root
        data = [int(x) for x in data.split(',') if x]
        return redeserialize()
        
        

# Your Codec object will be instantiated and called as such:
# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans
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

