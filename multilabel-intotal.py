import pandas as pd

# CSV文件的路径
file_path = r'D:\\EdgeDownload\\archive\\multilabel_modified\\multilabel_classification(2).csv'

# 读取CSV文件
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
    exit(1)
except pd.errors.EmptyDataError:
    print("文件是空的。")
    exit(1)
except pd.errors.ParserError:
    print("文件解析错误。请检查CSV格式。")
    exit(1)

# 确定类别列。假设前两列不是类别，从第3列到第18列（C到R）
# 如果类别列有特定的列名，可以根据列名来选择
category_columns = df.columns[2:18]

# 检查类别列是否存在
if len(category_columns) == 0:
    print("未找到类别列。请检查CSV文件的列布局。")
    exit(1)

# 确保类别列的数据是数值类型（0或1）
df[category_columns] = df[category_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# 计算每行中类别列为1的数量
df['num_categories'] = df[category_columns].sum(axis=1)

# 统计包含多个类别的图片数量
num_images_multiple_categories = (df['num_categories'] > 1).sum()

# 输出结果
total_images = len(df)
print(f"总共有 {total_images} 张图片。")
print(f"其中包含多个类别的图片数量为: {num_images_multiple_categories}")
