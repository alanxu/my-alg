https://torstencurdt.com/tech/posts/modulo-of-negative-numbers/

```python
# For Python
 3 % 10 = 3
-3 % 10 = 7

```

## Problems

## Number Conversion

### [12. Integer to Roman](https://leetcode.com/problems/integer-to-roman/)
```python
class Solution:
    

    def intToRoman(self, num: int) -> str:
        digits = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), 
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        
        roman_digits = []
        # Loop through each symbol.
        for value, symbol in digits:
            # We don't want to continue looping if we're done.
            if num == 0: break
            count, num = divmod(num, value)
            # Append "count" copies of "symbol" to roman_digits.
            roman_digits.append(symbol * count)
        return "".join(roman_digits)
```

### [273. Integer to English Words](https://leetcode.com/problems/integer-to-english-words/)

```python
class Solution(object):
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        
        if not num:
            return 'Zero'
        
        def trans(n):
            words = {
                0: 'Zero',
                1: 'One',
                2: 'Two',
                3: 'Three',
                4: 'Four',
                5: 'Five',
                6: 'Six',
                7: 'Seven',
                8: 'Eight',
                9: 'Nine',
                10: 'Ten',
                11: 'Eleven',
                12: 'Twelve',
                13: 'Thirteen',
                14: 'Fourteen',
                15: 'Fifteen',
                16: 'Sixteen',
                17: 'Seventeen',
                18: 'Eighteen',
                19: 'Nineteen',
                20: 'Twenty',
                30: 'Thirty',
                40: 'Forty',
                50: 'Fifty',
                60: 'Sixty',
                70: 'Seventy',
                80: 'Eighty',
                90: 'Ninety'
            }
            
            hundred = n // 100
            tens = n - hundred * 100
            
            
            result = ''
            
            if hundred:
                result = f'{words[hundred]} Hundred'
            
            if tens >= 20:
                ten = (n - hundred * 100) // 10
                rest = n - hundred * 100 - ten * 10
                result = f'{result} {words[ten*10]}'
                if rest:
                    result = f'{result} {words[rest]}'
            elif tens:
                result = f'{result} {words[tens]}'
            return result
        
        billion = num // 1000000000
        million = (num - billion * 1000000000) // 1000000
        thousand = (num - billion * 1000000000 - million * 1000000) // 1000
        rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000
        
        result = ''
        if billion:
            result = f'{trans(billion)} Billion'
        if million:
            result = f'{result} {trans(million)} Million'
        if thousand:
            result = f'{result} {trans(thousand)} Thousand'
        if rest:
            result = f'{result} {trans(rest)}'
            
        while('  ' in result):
            result = result.replace('  ', ' ')

        return result.strip()
```

## Others
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

### [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:

        c1, c2 = l1, l2
        carry = 0
        r, rc = None, None
        while True:
            if not c1 and not c2 and not carry:
                break
            v1, v2 = c1.val if c1 else 0, c2.val if c2 else 0

            s = v1 + v2 + carry
            v = s % 10
            carry = s //10
            
            if not r:
                r = rc = ListNode(v, None)
            else:
                rc.next = ListNode(v, None)
                rc = rc.next
                
            c1 = c1.next if c1 else None
            c2 = c2.next if c2 else None

        return r
```

### [8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/)

```python
class Solution:
    def myAtoi(self, str: str) -> int:
        str = str.strip()
        str = re.findall('(^[\+\-0]*\d+)\D*', str)

        try:
            result = int(str[0])
            MAX_INT = 2147483647
            MIN_INT = -2147483648
            if result > MAX_INT > 0:
                return MAX_INT
            elif result < MIN_INT < 0:
                return MIN_INT
            else:
                return result
        except:
            return 0
```

### [29. Divide Two Integers](https://leetcode.com/problems/divide-two-integers/)

### [43. Multiply Strings](https://leetcode.com/problems/multiply-strings/)

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        # From 2 strs, get each digit and its degree.
        # The product is the sum of product of each digit
        # with their correct degree
        
        ans = 0
        for i1, n1 in enumerate(num1[::-1]):
            for i2, n2 in enumerate(num2[::-1]):
                ans += int(n1) * (10 ** i1) * int(n2) * (10 ** i2)
        
        return str(ans)
```

### [50. Pow(x, n)](https://leetcode.com/problems/powx-n/)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            return self.myPow(1/x, -n)
        if n == 0: return 1
        half = self.myPow(x, n // 2)
        return half * half * (x if n & 1 else 1)
```

### [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/)

### [445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        n1, n2 = 0, 0
        cur1, cur2 = l1, l2
        
        while cur1:
            n1 += 1
            cur1 = cur1.next
        while cur2:
            n2 += 1
            cur2 = cur2.next
            
        cur1, cur2 = l1, l2
        head = None
        while n1 > 0 and n2 > 0:
            val = 0
            if n1 >= n2:
                val += cur1.val
                cur1 = cur1.next
                n1 -= 1
            if n1 < n2:
                val += cur2.val
                cur2 = cur2.next
                n2 -= 1
                
            cur = ListNode(val)
            cur.next = head
            head = cur
            
        cur, head = head, None
        carry = 0
        while cur:
            val = (cur.val + carry) % 10
            carry = (cur.val + carry) // 10 
            
            node = ListNode(val)
            node.next = head
            head = node
            
            cur = cur.next
            
        if carry:
            node = ListNode(carry)
            node.next = head
            head = node
            
        return head
```

### [66. Plus One](https://leetcode.com/problems/plus-one/)

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        N = len(digits)
        carry = 1
        for i in range(N - 1, -1, -1):
            carry, value  = divmod(digits[i] + carry, 10)
            digits[i] = value
            if not carry:
                break
        return digits if not carry else [carry] + digits
    
    def plusOne(self, digits: List[int]) -> List[int]:
        if digits == [9]:
            return [1, 0]
        if digits[-1] == 9:
            digits[-1] = 0
            digits[:-1] = self.plusOne(digits[:-1])
        else:
            digits[-1] += 1
        return digits
```