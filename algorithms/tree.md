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

## Others

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
            if data[0] == 'null':
                data.pop(0)
                return None

            root = TreeNode(data[0])
            data.pop(0)
            root.left = rdeserialize(data)
            root.right = rdeserialize(data)
            return root
        return rdeserialize(data.split(','))

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
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

### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def find_lca(root, p, q):
            """
            Given a node, find the LCA for p and q if they both under node. If only one under node,
            return p/q directly.
            KEY is: There is one and only one node for p and q, that p is in its left and q is in its right
                    if p and q are not directly adjacent. This node is the LCA.
                    And if p or q are each others acestor, then the lower one is the LCA.
            For each current node in func, There are several cases
            - The node is the LCA, or LCA is in node's left or right child (we can find LCA)
            - The node is in left or right child of LCA. (we can find p or q or None)
            """
            if root == p or root == q or not root:
                return root
            
            left = find_lca(root.left, p, q)
            right = find_lca(root.right, p, q)
            
            # We just need to know has any or has none
            return root if left and right else left or right
        
        return find_lca(root, p, q)
```

### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        self.pre = float('-inf')
        def traverse(node):
            if not node:
                return True

            if not traverse(node.left):
                return False
            
            if node.val <= self.pre:
                return False
            
            self.pre = node.val
            
            if not traverse(node.right):
                return False

            return True
        return traverse(root)
```


### [863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

```python
def distanceK(self, root, target, K):
    conn = collections.defaultdict(list)
    def connect(parent, child):
        # both parent and child are not empty
        if parent and child:
            # building an undirected graph representation, assign the
            # child value for the parent as the key and vice versa
            conn[parent.val].append(child.val)
            conn[child.val].append(parent.val)
        # in-order traversal
        if child.left: connect(child, child.left)
        if child.right: connect(child, child.right)
    # the initial parent node of the root is None
    connect(None, root)
    # start the breadth-first search from the target, hence the starting level is 0
    bfs = [target.val]
    seen = set(bfs)
    # all nodes at (k-1)th level must also be K steps away from the target node
    for i in range(K):
        # expand the list comprehension to strip away the complexity
        new_level = []
        for q_node_val in bfs:
            for connected_node_val in conn[q_node_val]:
                if connected_node_val not in seen:
                    new_level.append(connected_node_val)
        bfs = new_level
        # add all the values in bfs into seen
        seen |= set(bfs)
    return bfs
```

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

### [333. Largest BST Subtree](https://leetcode.com/problems/largest-bst-subtree/)

```python
class Solution:
    def largestBSTSubtree(self, root: TreeNode) -> int:
        # Recursively calculate/compare current node and based on its left/right subtree
        # KEY is to know if the left/right subtree is valid BST, and if yes, what is their min/max
        # Use the returned nums to indicate if the current node's left/right subtree is valide BST
        #  ==0: no subtree, valid, consider node itself.
        #  >0: there are subtree, valid, consider subtree
        #  -1: subtree invalid, so current node as root is invalid
        
        # When both left/right are valid BST, when len(left/right) > 0, need to compare the lmax 
        # and rmin. If all valid, calculate current min, max and n. 
        self.ans = 0
        def find(node):
            if not node:
                return None, None, 0
            
            lmin, lmax, lnum = find(node.left)
            rmin, rmax, rnum = find(node.right)
            
            left_valid, right_valid, curmin, curmax = False, False, None, None
            
            if lnum >= 0:
                left_valid = node.val > lmax if lnum > 0 else True
                curmin = lmin if lnum > 0 else node.val
            
            if rnum >= 0:
                right_valid = node.val < rmin if rnum > 0 else True
                curmax = rmax if rnum > 0 else node.val
                
            if left_valid and right_valid:
                n = lnum + rnum + 1
                self.ans = max(self.ans, n)
                return curmin, curmax, n
            else:
                return None, None, -1
            
        find(root)
        return self.ans
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

### [426. Convert Binary Search Tree to Sorted Doubly Linked List](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        first, last = None, None
        def dfs(node):
            # Trick: Use nonlocal
            nonlocal first, last
            if not node:
                return
            dfs(node.left)
            # new_node = Node(node.val)
            if not last:
                first = node
            else:
                node.left = last
                last.right = node
            last = node
            dfs(node.right)
        
        if not root:
            return root
        
        dfs(root)
        
        first.left = last
        last.right = first
        return first
```

### [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

```python
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
```

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
```
```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        self.previous_right = None
        def helper(root = root):
            if root:
                helper(root.right)
                helper(root.left)
                root.right, self.previous_right = self.previous_right, root
                root.left = None
        helper()
```

### [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        # Utilize feature of perfect binary tree.
        # On interate process one level, current interate is changing
        # next level.
        # Leftmost needs to be maintained to find next level
        
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

### [222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)

```python
class Solution:
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
```
```python
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        return 1 + self.countNodes(root.right) + self.countNodes(root.left) if root else 0
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

### [536. Construct Binary Tree from String](https://leetcode.com/problems/construct-binary-tree-from-string/)

```python
# Fav
class Solution:
    def str2tree(self, s: str) -> TreeNode:
        def build(i=0):
            root_val = []
            root = None
            while i <= len(s) - 1:
                if s[i] in ('(', ')'):
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

            return root if root else TreeNode(''.join(root_val)) 
        
        
        if not s:
            return None
        
        return build()
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
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        def dfs(node1, node2):
            
            if not node1 and not node2:
                return True
            
            if (not node2) or (not node1) or (node1.val != node2.val):
                return False
            
            return dfs(node1.left, node2.left) and dfs(node1.right, node2.right)
        
        return dfs(p, q)
```

### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
                
        return dp[-1]
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
```

### [1110. Delete Nodes And Return Forest](https://leetcode.com/problems/delete-nodes-and-return-forest/)

```python
class Solution:
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

### [103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        ret = []
        level_list = deque()
        if root is None:
            return []
        # start with the level 0 with a delimiter
        node_queue = deque([root, None])
        is_order_left = True

        while len(node_queue) > 0:
            curr_node = node_queue.popleft()

            if curr_node:
                if is_order_left:
                    level_list.append(curr_node.val)
                else:
                    level_list.appendleft(curr_node.val)

                if curr_node.left:
                    node_queue.append(curr_node.left)
                if curr_node.right:
                    node_queue.append(curr_node.right)
            else:
                # we finish one level
                ret.append(level_list)
                # add a delimiter to mark the level
                if len(node_queue) > 0:
                    node_queue.append(None)

                # prepare for the next level
                level_list = deque()
                is_order_left = not is_order_left

        return ret
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

