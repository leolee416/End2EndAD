import os
import re

def rename_files_in_current_and_subfolders():
    # 获取当前工作目录
    folder_path = os.getcwd()
    print(f"Working directory: {folder_path}")

    # 遍历当前文件夹及其子文件夹中的所有内容
    for root, _, files in os.walk(folder_path):
        for filename in files:
            print(f"Checking file: {filename}")  # 调试输出：当前文件

            # 构造完整的文件路径
            full_path = os.path.join(root, filename)

            # 如果文件名已经符合要求（_xxx 三位数字，且有附加内容），跳过
            if re.match(r"\d{8}_\d{6}_\d{3}.*", filename):
                print(f"File already matches the format: {filename}, skipping.")
                continue

            # 匹配文件名的时间戳和附加内容
            match = re.match(r"(\d{8}_\d{6})(.*)", filename)
            if match:
                # 提取时间戳和后续内容
                timestamp = match.group(1)
                extra_content = match.group(2)

                # 构造新的文件名
                new_filename = f"{timestamp}_000{extra_content}"
                new_full_path = os.path.join(root, new_filename)

                # 重命名文件
                print(f"Renaming {full_path} to {new_full_path}")  # 调试输出：重命名日志
                os.rename(full_path, new_full_path)

# 调用函数
rename_files_in_current_and_subfolders()
