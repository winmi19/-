from collections import Counter

s=input()
d=Counter(s)

print(d)
res=True
for x in d.values():
    if x!=1:
        res=False

print(res)