# 哈希表和前缀和

# 两数之和
def Twosum(nums:list[int],target:int)->int:
    n=len(nums)
    res=[]
    for i in range(n):
        for j in range(n):
            if i!=j and nums[i]+nums[j]==target:
                res.append([i,j])
    
    return res                                               #？？？？？????????

# def Twosum2(nums:list[int],target:int)->int:
#     d={}
#     n=len(nums)
#     res=[]
#     for i in range(n):
        
#         if d.get(target-nums[i]) is not None:
#             res.append([i,d[target-nums[i]]])
#         d[nums[i]]=i

#     return res

        


nums=list(map(int,input().split()))
res=Twosum(nums,4)
print(res)

res=Twosum2(nums,4)
print(res)

# d={'xu':0,'wen':1 , 'min':2}
# print(d.get('xu'))
# print(d.get('wen'))
# print(d.get('miao'))
# print(d['xu'])       # d['xu']和d.get('xu')的区别是如果键值不存在，前者报错，后者出'None'



# 一维的前缀和   给定q组的前缀和
#   前缀和p[0]=0   p[1]=a[0]    p[n]=a[0]+....a[n-1]
#  要求a[l]...a[r]   l<r
# p[l]=a[0]+...+a[l-1]    p[r+1]=a[0]+..a[r]

# n=int(input())  #数字个数
# a=list(map(int,input().split()))  #数组
# q=int(input())   #组的个数

# p=[0]*(n+1)
# # print(p)
# # print(type(p))
# for i in range(n):
#     p[i+1]=p[i]+a[i]   # 前缀和数组


# for _ in range(q):
#     l,r=map(int,input().split())
#     res=p[r+1]-p[l]
#     print(res)


