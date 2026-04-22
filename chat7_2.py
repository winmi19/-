# # 是否存在路径？   dfs方法
# from collections import defaultdict
# def findpath(edge:list[list[int]],source,destination)->bool:
#     #存储   [0,1][0,2][1,2]
#     g=defaultdict(list)
#     for x in edge:
#         g[x[0]].append(x[1])
#         g[x[1]].append(x[0])   #相互添加
    
 
#     def dfs(i):
#         if i==destination:
#             return True
#         vis.add(i)
#         for j in g[i]:
#             if j not in vis and dfs(j):
#                 return True
#         return False
            
#     vis=set()
#     return dfs(source)


# n=int(input())
# edge=[]
# for x in range(n):
#     lis=list(map(int,input().split()))
#     edge.append(lis)
# sre,des=map(int,input().split())
# print(findpath(edge,sre,des))


#dijkstra
