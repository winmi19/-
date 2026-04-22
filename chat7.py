# 并查集

n,m=map(int,input().split())      #n个人，m个关系，p次询问
fa=list(range(n+1))
def find(x):
    if fa[x]==x :return x
    fa[x]=find(fa[x])
    return fa[x]

def union(u,v):
    if find(u)!=find(v):
        fa[find(v)]=find(u)


for _ in range(m):
    u,v=map(int,input().split())
    union(u,v)

# for _ in range(p):
#     u,v=map(int,input().split())
#     if find[u]==find[v]:
#         print('yes')
#     else:
#         print('no')



# 村村通代码，先压缩成菊花集，然后cnt-1,联通集合减一
for x in range(1,n+1):
    fa[x]= find(x)

s=set(fa)
cnt=len(s)-1
print(cnt-1)

