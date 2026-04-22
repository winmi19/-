#分解因子
import math

def factor(n):
    
    factor=[]
    for i in range(1,int(math.sqrt(n)+1)):
        if n%i==0:
            factor.append(i)
            if n//i!=i:
                factor.append(n//i)
    
    factor.sort()
    return factor

print(factor(12))