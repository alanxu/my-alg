https://torstencurdt.com/tech/posts/modulo-of-negative-numbers/

```python
# For Python
 3 % 10 = 3
-3 % 10 = 7

```

## Problems

### [592. Fraction Addition and Subtraction](https://leetcode.com/problems/group-shifted-strings/)

```python
class Solution:
    def fractionAddition(self, expression):
        # Trick: Use regex to split string
        ints = map(int, re.findall('[+-]?\d+', expression))
        
        # A is numerator, B is denominator
        A, B = 0, 1
        for a in ints:
            # Trick: use next()
            b = next(ints)
            
            A = A * b + B * a
            B *= b
            
            gcd = math.gcd(A, B)
            
        A //= gcd
        B //= gcd
            
        return f"{A}/{B}"
```

### [356. Line Reflection](https://leetcode.com/problems/line-reflection/)
```python
class Solution:
    def fractionAddition(self, expression):
        # Trick: Use regex to split string
        ints = map(int, re.findall('[+-]?\d+', expression))
        
        # A is numerator, B is denominator
        A, B = 0, 1
        for a in ints:
            # Trick: use next()
            b = next(ints)
            
            A = A * b + B * a
            B *= b
            
            gcd = math.gcd(A, B)
            
        A //= gcd
        B //= gcd
            
        return f"{A}/{B}"
```

### [1344. Angle Between Hands of a Clock](https://leetcode.com/problems/angle-between-hands-of-a-clock/)
```python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        pos_hour, pos_min = (hour + minutes / 60) * 5, minutes
        
        distance = abs(pos_hour - pos_min)
        
        # Trick: use min to choose smaller angle
        return 360 * (min(distance, 60 - distance) / 60)
```

### [365. Water and Jug Problem](https://leetcode.com/problems/water-and-jug-problem/)
```python
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        # limit brought by the statement that water is finallly in one or both buckets
        if x + y < z:
            return False
        
        # case x or y is zero
        if x == z or y == z or x + y == z:
            return True
        
        # get GCD, then we can use the property of Bézout's identity
        return z % math.gcd(x, y) == 0
```

### [166. Fraction to Recurring Decimal](https://leetcode.com/problems/fraction-to-recurring-decimal/)
```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        # https://youtu.be/O04JAIm5HXY
        sign = '-' if numerator * denominator < 0 else ''
        numerator, denominator = abs(numerator), abs(denominator)
        
        # Initial calculate integer part
        n, remainder = divmod(numerator, denominator)
        
        ans = [str(sign), str(n)]
        
        if remainder:
            ans.append('.')
        
        hashmap = {}
        
        while remainder and remainder not in hashmap:
            hashmap[remainder] = len(ans)
            n, remainder = divmod(10 * remainder, denominator)
            ans.append(str(n))
            
        if remainder in hashmap:
            # Trick: Use insert()
            ans.insert(hashmap[remainder], '(')
            ans.append(')')
        return ''.join(ans)
```

### [368. Largest Divisible Subset](https://leetcode.com/problems/largest-divisible-subset/)
```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        # The inuition: if a % b = 0 and b % c = 0, then a % c = 0
        
        # Trick: Sort the array for combination questions to reduce times of iteration
        nums = sorted(nums)
        
        # Dict key is max num in that subset, value is the divisible subset
        subsets = {-1: set()}
        
        for num in nums:
            subsets[num] = max([subsets[k] for k in subsets if num % k == 0], key=len) | {num}
            
        return max([subsets[k] for k in subsets], key=len)
```

### [1497. Check If Array Pairs Are Divisible by k](https://leetcode.com/problems/check-if-array-pairs-are-divisible-by-k/)
```python
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        # https://leetcode.com/problems/check-if-array-pairs-are-divisible-by-k/discuss/709691/Java-7ms-Simple-Solution
        remainders = [0] * k
        for a in arr:
            rem = a % k
            if rem < 0:
                rem += k
            remainders[rem] += 1
        
        if remainders[0] % 2 != 0: return False
        
        # It cannot check k // 2 when k is even, so check 0 and others
        for i in range(1, k // 2 + 1):
            if remainders[i] != remainders[k - i]:
                return False
            
        return True
```

### [829. Consecutive Numbers Sum](https://leetcode.com/problems/consecutive-numbers-sum/)

```python
class Solution:
    def consecutiveNumbersSum(self, N: int) -> int:
        # N = (x + 1) + (x + 2) + ... + (x + k)
        # N is given, k is a range can be iterated, we need to know x for each k
        # We need to know two things: 1) What is the limit of k, 2) How to get x from k
        # For question 1), we change another variable x in to a scope x >= 0, then we can know k limit
        # For question 2), we use formular
        
        count = 0
        upper_limit = floor(math.sqrt(2 * N + 0.25) - 0.5) + 1
        for k in range(1, upper_limit):
            # Trick: Check float is integer
            if (N / k - (k + 1) / 2).is_integer():
                count += 1
        return count
```