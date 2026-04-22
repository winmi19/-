from collections import Counter


nums=list(map(int,input().split()))

d=Counter(nums)  #counter是dict的子类

print(d.items())
print(d.keys())
print(d.values())

sort_value=sorted(d.items(),key=lambda x:x[1])
print(d)

c=dict(d)
print(c)

