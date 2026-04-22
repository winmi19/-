
from collections import deque

def bfs(g:list[list[int]],i,j):  #找二维迷宫最短路劲
    n=len(g)
    m=len(g[0])  #列表的长宽
    q=deque()
    q.append([i,j,0])
    d=[(1,0),(-1,0),(0,1),(0,-1)]
    vis=set()
    vis.add((i,j))

    while q:
        x1,y1,steps=q.popleft()    #popleft是队列，Pop是栈
        if x1==m-1 and y1==n-1:
                return steps
        for nx,ny in d:
            x=nx+x1
            y=ny+y1

            

            if 0<=x<m and 0<=y<n and g[x][y]==1 and (x,y) not in vis:
                vis.add((x,y))
                q.append((x,y,steps+1))

    return -1





