import os
import shutil


def copy_clean_directories(source_dir, target_dir):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录下的所有子目录
    for subdir, dirs, _ in os.walk(source_dir):
        # 检查是否存在名为'clean'的目录
        if 'clean' in dirs:
            # 构建源'clean'目录的完整路径
            source_clean_dir = os.path.join(subdir, 'clean')

            # 计算源目录相对于源目录根目录的路径
            relative_path = os.path.relpath(subdir, source_dir)

            # 构建目标目录的完整路径，保持原有的目录结构
            target_dir_path = os.path.join(target_dir, relative_path)

            # 如果目标目录不存在，则创建它
            if not os.path.exists(target_dir_path):
                os.makedirs(target_dir_path)

            # 构建目标'clean'目录的完整路径
            target_clean_dir = os.path.join(target_dir_path, 'clean')
            os.makedirs(target_clean_dir,exist_ok=True)
            # 复制'clean'目录及其内容到目标目录
            for item in os.listdir(source_clean_dir):
                s = os.path.join(source_clean_dir, item)
                t = os.path.join(target_clean_dir, item)
                
                if os.path.isdir(s):
                    shutil.copytree(s, t, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, t)

# 使用示例
source_directory = '/home/kai/gamecheat/dataset/cs2_demo'  # 源文件夹路径
target_directory = '/home/kai/gamecheat/ae/data/cs2_demo'  # 目标文件夹路径

copy_clean_directories(source_directory, target_directory)