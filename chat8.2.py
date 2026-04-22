# # dp动态规划  网格dp

# #珠宝最大价值
# frame=[[1,3,1],[1,5,1],[4,2,1]]

# def maxvalue(frame:list[list[int]]):
#     m=len(frame)
#     n=len(frame[0])
    
#     f=[[0]*(n+1) for _ in range(m+1)]

#     for i in range(1,m+1):
#         for j in range(1,n+1):
#             v=frame[i-1][j-1]
#             f[i][j]=v+max(f[i-1][j],f[i][j-1])

#     print(f[m][n])

# maxvalue(frame)


#最小路径和
from math import inf
def minroute(g:list[list[int]]):
    m=len(g)
    n=len(g[0])

    f=[[inf]*(n+1) for _ in range(m+1)]
    f[0][1]=f[1][0]=0
    for i in range(1,m+1):
        for j in range(1,n+1):
            v=g[i-1][j-1]
            f[i][j]=v+min(f[i-1][j],f[i][j-1])

    return f[m][n]



#最大值问题 → 外界要“很小”（不被选）
#最小值问题 → 外界要“很大”（不被选）