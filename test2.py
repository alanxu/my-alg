import random

data = []
with open('friends.csv', 'r') as f:
    for line in f:
        line = line.strip().split(',')
        data.append({"name": line[0], "email": line[1]})

data = data[1:]


people_map = {i: data[i] for i in range(len(data))}
people_have_gift = set()
ans = []
for i, people in people_map.items():
    print(people)
    while True:
        idx = random.randint(0, len(data) - 1)
        p = people_map[idx]
        # print(p)
        if p['name'] in people_have_gift or idx == i:
            continue
        else:
            print(people)
            ans.append([people['email'], people_map[idx]['email']])
            break

print(ans)
