octet sequence



* Set union A | B
* Set intersection A & B
* Set subtraction A & ~B
* Set negation ALL_BITS ^ A or ~A
* Set bit A |= 1 << bit
* Clear bit A &= ~(1 << bit)
* Test bit (A & 1 << bit) != 0
* Extract last bit A&-A or A&~(A-1) or x^(x&(x-1))
* Remove last bit A&(A-1)
* Get all 1-bits ~0

https://leetcode.com/problems/sum-of-two-integers/discuss/84278/A-summary%3A-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently


mask = 0xFFFFFFFF
32 digit all 1

mask = 0x7FFFFFFF
In 2's complements, max positive only first is 0

x | 1
set 1 in smallest bit of x

```python
def is_odd(n):
    return n & 1 != 0
```

### [957. Prison Cells After N Days](https://leetcode.com/problems/prison-cells-after-n-days/)

```python
class Solution:
    def prisonAfterNDays1(self, cells: List[int], N: int) -> List[int]:
        pre, cur = cells, [0] * 8
        
        for _ in range(N):
            cur = [0] * 8
            for i in range(1, 7):
                cur[i] = 1 if pre[i - 1] == pre[i + 1] else 0
            pre = cur
        
        return cur
            
    def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:
        
        def next_day(state):
            state = ~ (state << 1) ^ (state >> 1)
            return state & 0x7e
        
        seen = {}
        fast_forwarded = False
        
        # Trick: Use 0x0 rather than 0, because it tells developer this is a bit mask and
        #        it is easier for computer to convert to binary.
        #        It is just different ways to represent a number, compiler will treat it same as 0
        state = 0x0
        
        for cell in cells:
            state <<= 1
            state = (state | cell)
            
        while N > 0:
            if not fast_forwarded:
                if state in seen:
                    N %= seen[state] - N
                    fast_forwarded = True
                else:
                    seen[state] = N
            
            # N might be 0 after fastforward
            if N > 0:
                N -= 1
                state = next_day(state)
        
        ans = []
        for _ in range(len(cells)):
            ans.append(state & 0x1)
            state = state >> 1
            
        return reversed(ans)
```

### [187. Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/)

```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        Wide, Len = 10, len(s)
        if Len < Wide:
            return []
        
        to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        nums = [to_int[c] for c in s]
        
        bitmask = 0x0
        seen, ans = set(), set()
        
        for start in range(Len - Wide + 1):
            if start == 0:
                for i in range(Wide):
                    bitmask <<= 2
                    bitmask |= nums[i]
            else:
                bitmask <<= 2
                bitmask |= nums[start + Wide -1]
                # Trick: Unset 2L and 2L + 1 bit
                bitmask &= ~(3 << 2 * Wide)
            if bitmask in seen:
                ans.add(s[start:start + Wide])
            seen.add(bitmask)
        return ans
```

### [898. Bitwise ORs of Subarrays](https://leetcode.com/problems/bitwise-ors-of-subarrays/)

```python
# Fav
# Difficult
class Solution:
    def subarrayBitwiseORs(self, arr: List[int]) -> int:
        # Trick: DP
        #   https://leetcode.com/problems/bitwise-ors-of-subarrays/discuss/165881/C%2B%2BJavaPython-O(30N)
        # Trick: State Compression
        #   Decrease demension - use ors[i] not or[i][j]
        #   Rolling Array - just need to save cur and pre state
        # or[i][j] = arr[i] | arr[i + 1] | ... | arr[j] (or[i][j] is a single value)
        # ors[i] = {or[0][i], or[1][i], ... or[i][i]}
        # ors[i] = {or[0][i - 1] | arr[i], or[1][i - 1] | arr[i], ..., or[i-1][i - 1] | arr[i], or[i][i]}
        # ors[i] = {x | arr[i] for x in ors[i - 1]} | arr[i]
        # result = {ors[0], ors[1], ... ors[i], ...}
        cur, ans = set(), set()
        for n in arr:
            cur = {x | n for x in cur} | {n}
            ans |= cur
        
        return len(ans)
```

### [201. Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/)

```python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        # All the binary num between m, n can be divided to parts
        # prefix + remaining
        # The prefix is same across all (m, n), an remaining are 
        # completely different on each pos for i and i + 1 if i is odd;
        # If m - n > 1, there is at least 1 case as above;
        # If m - n = 1, and m is odd, above case still apply;
        # If m - n = 1, and m is even, only last pos is changed to 1 for n, all
        # prefix is not changed, so case still apply
        # So, in remaining, there MUST be a 0 and 1 for each position.
        # So, the result is prefix000...
        # Based on the range of m, n, the prefix can vary (0 - m << 1)
        # when m - n = 1 and m is even, prefix is m << 1;
        # when n is significant than m, the prefix is 0, so result is 0
        shift = 1
        while m != n:
            m >>= 1
            n >>= 1
            shift <<= 1
        return m * shift
```


### [393. UTF-8 Validation](https://leetcode.com/problems/utf-8-validation/)

```python
class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        n_bytes = 0
        mask1, mask2 = 1 << 7, 1 << 6
        for num in data:
            if not n_bytes:
                # If n_bytes == 0, it is the first num, find n_bytes
                mask = 1 << 7
                # Trick: Change consecutive 1 using mask
                while num & mask:
                    mask >>= 1
                    n_bytes += 1
                
                # If n_bytes from first num is 0, it is single byte
                # continue to quit the loop and return True
                if n_bytes == 0:
                    continue
                
                # When single bytes, n_byte=0, when 2 bytes, n_bytes=2
                # so no possibility for 1. and max 4 bytes, so > 4 invalid
                if n_bytes == 1 or n_bytes > 4:
                    return False
            else:
                # If the first 2 digit are not '10', return False
                # Trick: Check is 1 or 0 using mask and &
                if not (num & mask1) or num & mask2:
                    return False
            n_bytes -= 1
        return n_bytes == 0
```

### [342. Power of Four](https://leetcode.com/problems/power-of-four/)

```python
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        # Trick: Check if num is a power of two: x > 0 and x & (x - 1) == 0
        # Trick: Check if even power: n & 0xaaaaaaaa (...010101010101)
        # Trick: Remove last 1: n & (n - 1)
        return n > 0 and n & (n - 1) == 0 and n & 0xaaaaaaaa == 0
```

### [268. Missing Number](https://leetcode.com/problems/missing-number/)

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # Trick: a^b^b = a
        # Trick: XOR is transferable
        # If no missing, i = num, then 0^0^1^1...n^n=0
        # if missing, then index 0, 1, 2, ... n - 1, and max(nums) = n
        # so if we add a n to index and make 0, 1, 2, .. n, the index
        # will be matching the nums and only one missing.
        # So XOR all index and value, will find the missing number
        missing = len(nums)
        for i, num in enumerate(nums):
            missing = missing ^ i ^ num
        return missing
```

### [371. Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/)

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        # Trick
        mask = 0xFFFFFFFF
        
        while b:
            base = (a ^ b) & mask
            carry = ((a & b) << 1) & mask 
            a, b = base, carry
        
        # Trick
        max_int = 0x7FFFFFFF
        return a if a < max_int else ~(a ^ mask)
```

### [421. Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)

```python
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        L = len(bin(max(nums))) - 2
        max_xor = 0
        for i in range(L)[::-1]:
            max_xor <<= 1
            # Set last bit
            cur_xor = max_xor | 1
            prefixes = {num >> i for num in nums}
            # a ^ b = cur_xor, a ^ b must = pre_xor two with less 1
            max_xor |= any(cur_xor ^ p in prefixes for p in prefixes)
        return max_xor
```