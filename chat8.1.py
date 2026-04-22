# # 01背包问题

# W,N=map(int,input().split())  #W总重量，N总数量
# #重量
# w=list(map(int,input().split()))
# #价值
# v=list(map(int,input().split()))

# f=[[0]*(W+1) for _ in range(N+1)]

# for i in range(1,N+1):
#     for j in range(1,W+1):
#         if j>=w[i]:
#             f[i][j]=max(f[i-1][j],f[i-1][j-w[i]]+v[i])  # 前一个物品选不选，j-w[i]是指
#                          #前一个商品在选择了这一个的情况下，最大价值
#         else:
#             f[i][j]=f[i-1][j]     #当前重量小于目前物品的重量

# print(f[W][N])


#爬楼梯问题
# def climb(n):
#     def dfs(i):
#         if i==0 or i==1:
#             return 1
#         dfs(i-1)+dfs(i-2)   #两步或者一步
#     return dfs(n)



#采药问题
#T,M 价值，草药总数目
#M行 t和m
#01背包
# T,M=map(int,input().split())
# t=[0]*(M+1)
# v=[0]*(M+1)
# for i in range(1,M+1):
#     t[i],v[i]=map(int,input().split())   #时间和价值

# f=[[0]*(T+1) for _ in range(M+1)]   #行是时间
# for i in range(1,M+1):
#     for j in range(1,T+1):
#         if j>=t[i]:  #时间够的情况下 ,选或者不选
#             f[i][j]=max(f[i-1][j],f[i-1][j-t[i]]+v[i])
#         else:
#             f[i][j]=f[i-1][j]   #不选目前这朵花

# print(f[M][T])     #易错点！！！



#最大限度提高水果口味  
n=int(input()) #数量
price=list(map(int,input().split()))  #每个价格
taste=list(map(int,input().split()))  #口味
maxAmount=int(input())     #最大价格
maxCoupons=int(input())    #最大优惠次数，半价购买

    


