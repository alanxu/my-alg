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
    def str2tree(self, s: str) -> Optional[TreeNode]:
        if not s:
            return None
        def dfs(start):
            i = start
            num, sign = 0, 1
            node = None
            while i < len(s):
                x = s[i]
                if x in '()':
                    if not node:
                        node = TreeNode(num * sign)
                        # num, sign = 0, 1
                    if x == '(':
                        if not node.left:
                            node.left, i = dfs(i + 1)
                        else:
                            node.right, i = dfs(i + 1)
                    elif x == ')':
                        return node, i + 1
                else:
                    if x == '-':
                        sign = -1
                    elif x.isdigit():
                        num = num * 10 + int(x)
                    i = i + 1
            return node if node else TreeNode(num * sign)
        return dfs(0)
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

    def tree2str(self, root: TreeNode) -> str:
        def dfs(node):
            res = str(node.val)
            
            if node.left:
                res += f'({dfs(node.left)})'

            if node.right:
                if not node.left:
                    res += '()'
                res += f'({dfs(node.right)})'
            return res
        return dfs(root)
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

### [108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def build(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = build(left, mid - 1)
            root.right = build(mid + 1, right)
            return root
        return build(0, len(nums) - 1)
```

### [1628. Design an Expression Tree With Evaluate Function](https://leetcode.com/problems/design-an-expression-tree-with-evaluate-function/)

```python
import abc 
from abc import ABC, abstractmethod 
"""
This is the interface for the expression tree Node.
You should not remove it, and you can define some classes to implement it.
"""

class Node(ABC):
    @abstractmethod
    # define your fields here
    def evaluate(self) -> int:
        pass

class ExpressionNode(Node):
    def __init__(self, op):
        self.left = None
        self.right = None
        self.op = op
    def evaluate(self) -> int:
        if self.left and self.right:
            left, right = self.left.evaluate(), self.right.evaluate()
            if self.op == '+':
                return left + right
            elif self.op == '-':
                return left - right
            elif self.op == '*':
                return left * right
            elif self.op == '/':
                return left // right
        else:
            return int(self.val)

class ValueNode(Node):
    def __init__(self, value):
        self.val = value
    def evaluate(self) -> int:
        return int(self.val)

"""    
This is the TreeBuilder class.
You can treat it as the driver code that takes the postinfix input
and returns the expression tree represnting it as a Node.
"""

class TreeBuilder(object):
    def buildTree(self, postfix: List[str]) -> 'Node':
        val = postfix.pop()
        root = None
        if val in '+-*/':
            root = ExpressionNode(val)
            root.right = self.buildTree(postfix)
            root.left = self.buildTree(postfix)
        else:
            root = ValueNode(int(val))
        return root
		
"""
Your TreeBuilder object will be instantiated and called as such:
obj = TreeBuilder();
expTree = obj.buildTree(postfix);
ans = expTree.evaluate();
"""
        
```
