import pandas as pd

# 读取的文件是DataFarame格式
train_file = pd.read_csv('D:/PycharmProjects/learning/tree/treeData/sample_submission.csv')
# 将文件中label这一列拿出来 把格式转换为numpy格式
data = train_file['label'].values
# 去除重复项，转换为list形式
data_list = list(set(data))
dict = {}
for i in range(len(data_list)):
    # 构造映射字典
    dict[data_list[i]] = i
# 利用映射将字符串类别转换为数字
train_file['label_number'] = train_file['label'].map(dict)
# 将修改的文件保存到本地 不保存序号
train_file.to_csv('D:/PycharmProjects/learning/tree/treeData/sample_submission_.csv', index=False)
