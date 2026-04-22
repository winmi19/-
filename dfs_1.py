#给定一个二维网格，0 表示水域，1 表示陆地，找出网格中岛屿的数量。

def dfs(g:list[list[int]],i,j):
    d=[(1,0),(-1,0),(0,1),(0,-1)]
    n=len(g)
    m=len(g[0])
    g[i][j]=0  #设置为0 不再遍历
    

    
    #bfs用队列，dfs用递归
    for dx,dy in d:
        nx=i+dx
        ny=j+dy
        if 0<=nx<n and 0<=ny<m and g[nx][ny]==1: 
            dfs(g,nx,ny)


n=int(input())
tot=[]
for _ in range(n):
    tot.append(list(map(int,input().split())))
m=len(tot[0])

cnt=0
for i in range(n):
    for j in range(m):
        if tot[i][j]==1:
            dfs(tot,i,j)
            cnt+=1