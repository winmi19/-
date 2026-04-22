#堆详情

from heapq import *


# heappush(hq,5)
# heappush(hq,9)
# heappush(hq,11)
# heappush(hq,12)
# heappush(hq,13)
# heappush(hq,15)

# print(hq[0])
# print(hq[5])
# print(hq)

# heappop(hq)   #括号里面要写
# print(hq)

nums=[15,13,9,5,11,12]
hq=[]
for x in nums:
    heappush(hq,-x)

print(-hq[0])
print(hq)

#弹出最大
heappop(hq)
print(-hq[0])
print(hq)