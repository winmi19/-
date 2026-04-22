# #并发集
#查询是否有亲戚关系
# n个人，m个关系，p个查询
n,m,p=map(int,input().split())

fa=list(range(n+1))

#查找
def find(x):
    if fa[x]==x:
        return x
    fa[x]=find(fa[x])
    return fa[x]  #压缩成菊花集

def union(u,v):
    if find(u)!=find(v):
        fa[find(u)]=find(v)



for _ in range(m):
    u,v=map(int,input().split())
    union(u,v)

res=[]
for _ in range(p):
    u,v=map(int,input().split())
    if find(u)==find(v):
        res.append(1)
    else:
        res.append(0)
    
print(res)
for x in res:
    if x==1:
        print('yes')
    else:
        print('no')






