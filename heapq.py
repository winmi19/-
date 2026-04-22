# from heapq import *

#前k个高频数字：
#字典法
from collections import Counter
nums=list(map(int,input().split()))

d={}
# for x in nums:
#     d[x]=nums.count(x)  #这样写复杂度是N方
# print(d)
d=Counter(nums)

print(d)
print(type(d))


