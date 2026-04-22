#找约数

from math import *
def solve(n):
    res=[]
    for x in range(2,int(sqrt(n))+1):
        if n%x==0:
            res.append(x)
            if n//x !=x:
                res.append(n//x)
    return res

print(solve(12))    #真约数

#约数 1 2 3 4 6 12
#真约数 1 2 3 4 6 没有本身


#质因数分解
from math import *
def solve(n):
    d={}
    for i in range(2,int(sqrt(n))+1):
        while n%i==0:
            n//=i
            d[i]=d.get(i,0)+1
        
    if n>1:
        d[n] = d.get(n, 0) + 1

    return d



            
    