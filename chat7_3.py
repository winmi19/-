#动态规划

#爬楼问题
# def climb(n:int):  #n个台阶
#     if n==0 or n==1 :
#         return 1
#     return climb(n-1)+climb(n-2)
   

# print(climb(3))


#打家劫舍
# def rob(nums:list[int]):   #n个房屋
#     n=len(nums)
#     f=[[0]*2 for _ in range(n+1)]  #0-n  n+1个
#     for i in range(1,n+1):
#         f[i][0]=max(f[i-1][0],f[i-1][1])      #0代表目前这家不打劫
#         f[i][1]=f[i-1][0]+nums[i-1]        #第i家，nums的索引减1
#     return max(f[n][0],f[n][1])

# print(rob([1,2,3,1]))
# print(rob([2,7,9,3,1]))


#01背包问题  要不要的问题，两列
#找最大价值
#N个物品，W重量
# def bag(w:list[int],v:list[int],W:int):
#     n=len(w)   #个数
#     f=[[0]*(W+1) for _ in range(n+1)]
#     for i in range(1,n+1):
#         for j in range(1,W+1):
#             if j-w[i-1]>=0:
#                 f[i][j]=max(f[i-1][j],f[i-1][j-w[i-1]]+v[i-1])
#             else:
#                 f[i][j]=f[i-1][j]
#     print(f[n][W])
# #f[][]从0-n（共n+1个）    w,v只有从0-(n-1)  （共n个）


# w = [1, 3, 4]
# v = [15, 20, 30]
# W = 4

# bag(w,v,4)


#采集药物问题  限制是t,希望v达到最大   列数代表选不选，行数代表每株植物的判断
def collection(T:int,M:int,t:list[int],v:list[int]): #总时间限制，数目，每株时间，每株价格
    f=[[0]*(T+1) for _ in range(M+1)]
    for i in range(1,M+1):
        for j in range(1,T+1):
            if j-t[i-1]>=0:
                f[i][j]=max(f[i-1][j],f[i-1][j-t[i-1]]+v[i-1])
            else:
                f[i][j]=f[i-1][j]
    print(f[M][T])


#最大限度提高水果味道  t最大，限制价格
def  fruit(price:list[int],tastiness:list[int],n:int,amount:int,coupon:int):
    f=[[0]*(amount+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,amount+1):
            if j-price[i-1]>=0:
                f[i][j]=max(f[i][j])





