# 根据升高重建队伍

# def change(lis:list[list[int]])->list[list[int]]:
#     #[7,0],[7,1]
#     lis.sort(key=lambda x:(-x[0],x[1]))
#     print(lis)

#     res=[]
#     for i,p in enumerate(lis):
#         h,k=p[0],p[1]
#         if i==k:
#             res.append(p)
#         elif k<i:
#             res.insert(k,p)
#     print(res)


# n=int(input('有多少人'))
# people=[]
# for _ in range(n):
#     temp=[]
#     temp=list(map(int,input().split(',')))
#     people.append(temp)

# print(people)

# #people = [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
# change(people)


#分糖果问题：
n=int(input())
nums=list(map(int,input().split(',')))
#每一轮把糖果分给左手边的小孩

tol=sum(nums)

cut=[0]*n

while True:
    cut=nums.copy()
    for i ,x in enumerate(nums):
        cut[i]=(x+nums[(i+1)%n])//2
        if cut[i] & 1:cut[i]+=1

    nums=cut
    if len(set(cut))==1:
        break

print(sum(nums)-tol)


