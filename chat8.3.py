#完全背包问题

#总金额 t    钱币可能coin

def change(t:int,coin:list[int]):
    n=len(coin)

    f=[[0]*(t+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,t+1):
            c=coin[i-1]
            f[i][j]=f[i-1][j]   #没选这个钱币的时候的情况
            if j>=coin[i-1]:
                f[i][j]+=f[i][j-c]

    return f[n][t]
