
### [50. Pow(x, n)](https://leetcode.com/problems/powx-n/)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def my_pow(x, n):
            if n == 0:
                return 1.0

            half = my_pow(x, n//2)

            ans = half * half

            if n%2 == 1:
                ans = ans * x

            return ans

        if n < 0:
            n = -n
            x = 1/x

        return my_pow(x, n)
```

### [241. Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/)

```python
class Solution:
    #Trick: String parsing for math expression
    ops = {
        '+': lambda a, b: str(int(a) + int(b)),
        '-': lambda a, b: str(int(a) - int(b)),
        '*': lambda a, b: str(int(a) * int(b))
    }
    def diffWaysToCompute(self, input: str) -> List[int]:
        # Trick: Divide and Conquer
        #    Another resursion method, difference with others is it do own
        #    operation after the sub problem call. It depends on where is
        #    the complexity, then use resursion to get rid of it.
        ans = []
        for i in range(len(input)):
            ch = input[i]
            if ch in '+-*':
                ans_l = self.diffWaysToCompute(input[:i])
                ans_r = self.diffWaysToCompute(input[i + 1:])
                # This is faster
                # for l in ans_l:
                #     for r in ans_r:
                #         ans.append(self.ops[ch](l, r))
                ans.extend([self.ops[ch](l, r) for l in ans_l for r in ans_r])
        return ans if ans else [input]
```

### [218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

### [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

```python
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:

        def dist(i):
            return points[i][0] ** 2 + points[i][1] ** 2

        # https://www.youtube.com/watch?v=G9VcMTSZ1Lo&t=18s
        def partition(i, j):
            oi = i
            pivot = dist(i)
            i += 1

            while True:
                # Find the leftmost value that > pivot
                while i < j and dist(i) < pivot:
                    i += 1
                # Find the rightmost value < pivot
                while i <= j and dist(j) >= pivot:
                    j -= 1
                # If pointers meet, complete
                if i >= j: break
                # If pointers not meet, swap
                points[i], points[j] = points[j], points[i]

            # j is currently the last value < pivot
            points[oi], points[j] = points[j], points[oi]

            # After final swap, j become mid
            return j


        def sort(i, j, K):
            if i >= j: return

            mid = partition(i, j)

            len_left = mid - i + 1
            if len_left > K:
                sort(i, mid - 1, K)
            elif len_left < K:
                sort(mid + 1, j, K - len_left)

        sort(0, len(points) - 1, K)
        return points[:K]
```


### [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

### [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/)
```python
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        prefix_sum = [0]
        for x in nums:
            prefix_sum.append(prefix_sum[-1] + x)

        def mergeSort(l, r):
            # Sort prefix_sum of index [l:r] and count the num of pairs [i:j] where
            # lower <= prefix_sum[j] - prefix_sum[i] <= upper

            # we are acting or prefix sum, l == r means single sum not single element
            # in this case single element means [0:i]
            if l == r:
                return 0

            # devide and conquer, first select the mid index
            mid = l + (r - l)//2

            # count the value returned by each half
            cnt = mergeSort(l, mid) + mergeSort(mid + 1, r)

            # then need to count the value across 2 halves where i in 1sr half, j in
            # 2nd half
            j = jj = mid + 1
            for i in range(l, mid+1):
                while j <= r and prefix_sum[j] - prefix_sum[i] < lower: j += 1
                while jj <= r and prefix_sum[jj] - prefix_sum[i] <= upper: jj += 1
                # we know that the values are sorted in each half, so jj - j is the count
                cnt += jj - j

            # Need to merge the two halves, there might be better way, so the outter call
            # can correctly count the cross half value
            prefix_sum[l:r+1] = sorted(prefix_sum[l:r+1])
            return cnt

        # We perform in whole prefix_sum with first value 0, because we are looking for [i:j] i < j,
        # where [0:j] represent a single num in nums.
        return mergeSort(0, len(prefix_sum) - 1)

```


### [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

```python
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # Trick: Divid and Conquer
        # enum maitains original list order initially, mergeSort divid
        # the enum in 2 half, each half's order will be changed, but
        # the before-after relation of different half is not changed.
        # after mergeSort call to 2 half, each half is ordered and half 1
        # are all before half 2. Based on this, we can further calculate
        # num in half1 bigger than num in half2, not the opposite, because of
        # order of 2 half are fixed.

        res = [0] * len(nums)
        enum = list(enumerate(nums))

        def mergeSort(l, r):
            if l >= r:
                return

            mid = l + (r - l) // 2

            mergeSort(0, mid)
            mergeSort(mid + 1, r)

            # Use i, j pointing begining of half 1 and half 2
            i, j = l, mid + 1
            # inverse_count not only count for current i, it is accumulated num
            # of inverse from i = l
            inverse_count = 0
            while i <= mid and j <= r:
                if enum[i][1] <= enum[j][1]:
                    res[enum[i][0]] += inverse_count
                    i += 1
                else:
                    inverse_count += 1
                    j += 1

            # There might be bigger nums in half 1 bigger than all in half 1
            # whole num of half2 should be cunted to that num and onward in half1
            while i <= mid:
                res[enum[i][0]] += r - mid
                i += 1

            # Sort the whole segment in this recursion for use in outter call
            enum[l:r+1] = sorted(enum[l:r+1], key=lambda e:e[1])

        mergeSort(0, len(nums) - 1)
        return res

```
