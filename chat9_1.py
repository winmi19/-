from math import inf

n,m=map(int,input().split()) #n个节点  m条边
# #邻接表
# e=[[] for _ in range(n)]
# for _ in range(m):
#     u,v,p=map(int,input().split())
#     e[u].append((v,p))
#     e[v].append((u,p))

#邻接矩阵   无向有权
g=[[inf]*n for _ in range(n)]
for _ in range(m):
    u,v,p=map(int,input().split())
    g[u][v]=g[v][u]=p
    g[u][u]=g[v][v]=0



#Dijkstra
d=[inf]*n
d[0]=0    #到自己距离为0
s=set()

#遍历n-1轮，算其他几个到0的距离
for _ in range(n-1):
    x=-1
    for u in range(n):
        if u not in s and (x<0 or d[u]<d[x]):
            x=u
    s.add(x)
    
    #判断每个节点作为中间节点，最后的d[u]（每个节点到源点距离）最小值是否改变
    for u in range(n):
        d[u]=min(d[u],d[x]+g[u][x])




