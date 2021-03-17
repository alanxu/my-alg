
### [264. Ugly Number II](https://leetcode.com/problems/ugly-number-ii/)

```python
class Ugly:
    def __init__(self):
        # Trick: Not use self.x
        self.nums = nums = [1]
        # Trick: 3 Pointers
        i2 = i3 = i5 = 0
        for i in range(1, 1690):
            ugly = min(nums[i2] * 2, nums[i3] * 3, nums[i5] * 5)
            nums.append(ugly)
            
            if ugly == nums[i2] * 2:
                i2 += 1
            if ugly == nums[i3] * 3:
                i3 += 1
            if ugly == nums[i5] * 5:
                i5 += 1

class Solution:
    # Trick: Load on initiation
    u = Ugly()
    def nthUglyNumber(self, n: int) -> int:
        return self.u.nums[n - 1]
```

### [932. Beautiful Array]()

```python
class Solution:
    def beautifulArray(self, N: int) -> List[int]:
        res = [1]
        while len(res) < N:
            print(res)
            res = [i * 2 - 1 for i in res] + [ i * 2 for i in res]
        print(res)
        return [i for i in res if i <= N]
```

### [855. Exam Room](https://leetcode.com/problems/exam-room/)

```python
class ExamRoom:
    def __init__(self, N: int):
        self.N = N
        self.students = []

    def seat(self) -> int:
        if not self.students:
            # Trick: Dont't have to defin pos on top
            pos = 0
        else:
            # Trick: Use 0 for initial value of pos, dist, it
            #   magicaly avoid to handle the case before first student
            pos, dist = 0, self.students[0]
            
            # For each adjcent existing student, calculate dist if seat in
            # between, if the dist is larger, set pos for now
            for i in range(1, len(self.students)):
                _dist = (self.students[i] - self.students[i - 1]) // 2
                if _dist > dist:
                    pos, dist = self.students[i - 1] + _dist, _dist
            
            # Finaly, check seating after last student
            if dist < self.N - 1 - self.students[-1]:
                pos = self.N - 1

        # Trick: Use bisect to keep list sorted
        bisect.insort(self.students, pos)
        return pos
        

    def leave(self, p: int) -> None:
        # Trick: list.remove()
        self.students.remove(p)
        pass
```