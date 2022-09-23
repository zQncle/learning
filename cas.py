# 导入os模块
import os
import numpy as np
import pandas as pd
# path定义要获取的文件名称的目录
path = "C:\\Users\\Administrator\\Desktop\\产品"
# os.listdir()方法获取文件夹名字，返回数组
file_name_list = os.listdir(path)
file_name = []
for i in range(len(file_name_list)):
    file_name.append(file_name_list[i].split(" ")[0])
print(file_name)
A = np.array(file_name)
data = pd.DataFrame(A)
writer = pd.ExcelWriter('产品.xlsx')		# 写入Excel文件
data.to_excel(writer, 'cas号', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()