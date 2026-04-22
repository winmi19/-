#岛屿数量

from collections import deque

def bfs(g:list[list[int]],i,j):
    n=len(g)
    m=len(g[0])
    q=deque()
    q.append((i,j))
    d=[(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        x,y=q.popleft()
        g[x][y]=0
        for dx,dy in d:
            nx=x+dx
            ny=y+dy

            if 0<=nx<n and 0<=ny<m and g[nx][ny]==1: 
                g[nx][ny]=0  #bfs作用把周围都设置为0，
                q.append((nx,ny))


n=int(input())
tot=[]
for _ in range(n):
    tot.append(list(map(int,input().split())))
m=len(tot[0])

cnt=0
for i in range(n):
    for j in range(m):
        if tot[i][j]==1:
            bfs(tot,i,j)
            cnt+=1







