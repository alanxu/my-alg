


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


### [937. Reorder Data in Log Files](https://leetcode.com/problems/reorder-data-in-log-files/)

```python
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def get_key(log):
            _id, rest = log.split(" ", maxsplit=1)
            # Trick: sorted() remain same order when key are same
            return (0, rest, _id) if rest[0].isalpha() else (1, )
        return sorted(logs, key=get_key)
```