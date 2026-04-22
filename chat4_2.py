# def Two_sum(nums:list[int],t:int):
#  d={}
#  res=[]
#  for i,x in enumerate(nums):
#     if d.get(t-x) is not None:
#         res.append([i,d.get(t-x)])
#     d[x]=i  #字典里面存储索引，返回索引对

#  return res

# nums = [3, 2, 4, 3]
# t = 6
# res=Two_sum(nums,t)
# for x in res:
#    x.sort()
# res.sort()
# print(res)



#前缀和
#p[n]=a[0]+~~~+a[n-1]

def makep(a:list[int]):
    n=len(a)
    p=[0]*(n+1)
    for i in range(1,n+1):
        p[i]=p[i-1]+a[i-1]  #对a要往前移一个，单位索引
    return p

#具体应用：
#输出a[l]---a[r]
#p[r+1]-p[l]   (p[n]不包含a[n])