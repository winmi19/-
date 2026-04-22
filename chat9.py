
# n=4
# lis=[[] for _ in range(4)]
# lis[0].append(1)
# lis[0].append(2)
# lis[1].append(3)
# lis[2].append(4)
# print(lis)   #列表正确遍历方式


# 无向图邻接矩阵   有权无权取决于w有否
from math import inf
n,m=map(int,input().split())  # 点数，边数
g=[[inf]*n for _ in range(n)]

#邻接表   带权无向图
lis=[[] for _ in range(n)]


for _ in range(m):
    u,v,w=map(int,input().split())   # 两个点和他们之间的权重
    g[u][v]=g[v][u]=w
    g[u][u]=g[v][v]=0
    
    lis[u].append((v,w))
    lis[v].append((u,w))


    

print(g)
print(lis)

#带权有向图 lis[u].append((v,w)) 一句就好
#无权有向图  lis[u].append(v)

s=set()
#图的遍历  表
def dfs(u):
    #跑了这个节点，输出这个节点信号
    print(u,end=' ')
    set.add(u)

    for x in lis[u]:   #用邻接表
        if x not in s:
            dfs(x)







