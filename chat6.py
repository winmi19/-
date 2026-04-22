# 岛屿的最大面积

from collections import deque
def Maxarea(dao:list[list[int]])->int:
    n,m=len(dao),len(dao[0])
    res=0
    di=[(0,1),(0,-1),(1,0),(-1,0)]
    
    def bfs(i,j):
        ans=1
        q=deque([(i,j)])
        dao[i][j]=0
        while q:
            x,y=q.popleft()
            for dx,dy in di:
                nx=x+dx
                ny=y+dy
                if  0<=nx<n and 0<=ny<m and dao[nx][ny]==1 :
                    q.append((nx,ny))
                    ans+=1
                    dao[nx][ny]=0  #已经标记
        return ans
    

    
    for i,row in enumerate(dao):
        for j,x in enumerate(row):
            if x==1:
                res=max(res,bfs(i,j))
    return res


grid = [
[0,0,1,0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,0,1,1,1,0,0,0,0],
[0,1,1,0,1,0,0,0,0,0,0,0,0],
[0,1,0,0,1,1,0,0,1,0,1,0,0],
[0,1,0,0,1,1,0,0,1,1,1,0,0],
[0,0,0,0,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,0,0,0,0,0]
]

print(Maxarea(grid))




