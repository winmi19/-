# from bisect import *

# arr=[1,9,9,200,200]

# print(bisect(arr,1))  #找出第一个大于1的索引


# from bisect import *

# nums=list(map(int,input().split()))
# t=int(input())


# if t not in nums:
#     print(-1)
# else:
#     res=bisect(nums,t-1)
#     print(res)

from bisect import *

nums=list(map(int,input().split()))
t=int(input())
if t not in nums:
    res=[-1,-1]
    print(res)
else:
    xia=bisect(nums,t-1)
    shang=bisect(nums,t)
    res=[xia,shang-1]
    print(res)



