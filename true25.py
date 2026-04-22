# #优美字符串
# st='abcd'
# # print(st[:1])   #后一个取不到
# lis=list(st)
# print(lis)

# clip=''.join(lis)
# print(clip)

# st2='11233'
# lis2=list(st2)
# print(lis2)
# clip2=''.join(lis2)    #join里面必须要是字符串，或者字符串列表，转成字符串
# print(clip2)


# a=[1,2,4,5]
# clip3=' '.join(map(str,a))    #map是把其他东西转化为字符串
# print(clip3)


import sys
#优美字符串：
#出长度最大的优美字符串，如果存在多个答案，优先使用字典序最小的那一个作为答案。
words=[]

for line in sys.stdin:
    line=line.strip()  #去除空格和换行
    if line:   #如果不是空行
        words.append(line)

#把words里面的东西都排序
words.sort(key=(lambda x:(len(x),x)))  #第一看长度，第二看字符排序

good=set()
ans=' '
for w in words:
    if len(w)==1:
        ok=True
    else:
        ok =w[:-1] in good

    if ok:
        good.add(''.join(sorted(w)))  #进去的时候调整顺序，方便下一个判断
        if len(w)>len(ans) or (len(w)==len(ans) and w<ans):
            ans=w

print(ans)

            
        

