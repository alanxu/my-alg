## Problems

### [297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

```python
class Codec:

    def serialize(self, root):
        self.ans = ''
        def rserialize(n):
            if not n:
                self.ans += 'None, '
                return
            self.ans += f'{str(n.val)}, '
            rserialize(n.left)
            rserialize(n.right)
        
        rserialize(root)
        return self.ans

    def deserialize(self, data):
        #Trick: Tink about the exit condition
        def rdeserialize(l):
            # print(l)
            if l[0] == 'None':
                l.pop(0)
                return None
            r = TreeNode(l[0])
            l.pop(0)
            r.left = rdeserialize(l)
            r.right = rdeserialize(l)
            return r
        
        return rdeserialize(data.split(', '))
```

### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

```python
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
            # Trick: Exclude if the value is negative
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


