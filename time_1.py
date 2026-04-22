
from datetime import datetime,timedelta

#将输入转化为时间格式
s=input()   #"%y-%m-%d"
base=datetime.strptime(s,"%Y-%m-%d")

#将输出转化为特定字符串格式
out=datetime(2006,3,19)
res=out.strftime("%Y年%m月%d日")
print(res)
