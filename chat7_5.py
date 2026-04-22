#村村通：至少还要修多少条路

import sys

#查找
def find(x):
    if fa[x]==x:
        return x
    fa[x]=find(fa[x])
    return fa[x]  #压缩成菊花集

def union(u,v):
    if find(u)!=find(v):
        fa[find(u)]=find(v)

ans=[]
while  True:
    line=sys.stdin.readline().split()
    if not line or line[0]=='0':
        break

    n,m=int(line[0]),int(line[1])
    fa=list(range(n+1))
    
    for _ in range(m):
        u,v=map(int,input().split())
        union(u,v)

    for x in range(1,n+1):
        fa[x]=find(x)  #找到根节点  压成菊花集

    res=len(set(fa))-1  #目前有多少块联通集  -1是减去fa[0]
    ans.append(res-1)   #联通集数量-1就是路线数量

print(ans)


    