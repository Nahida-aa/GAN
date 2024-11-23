import csv
import re

def convert_log_to_csv(log_file_path, csv_file_path):
    # 定义正则表达式模式来匹配日志行
    log_pattern = re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO \[Epoch (?P<epoch>\d+)/\d+\] \[Batch (?P<batch>\d+)/\d+\] \[D loss: (?P<d_loss>[\d\.]+)\] \[G loss: (?P<g_loss>[\d\.]+)\]'
    )

    # 打开日志文件并读取内容
    with open(log_file_path, 'r') as log_file:
        log_lines = log_file.readlines()

    # 打开 CSV 文件并写入表头
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Timestamp', 'Epoch', 'Batch', 'D Loss', 'G Loss'])

        # 解析每一行日志并写入 CSV 文件
        for line in log_lines:
            match = log_pattern.match(line)
            if match:
                writer.writerow([
                    match.group('timestamp'),
                    match.group('epoch'),
                    match.group('batch'),
                    match.group('d_loss'),
                    match.group('g_loss')
                ])

    print(f"Log data has been converted to CSV format and saved to {csv_file_path}")
    
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 定义日志文件和输出 CSV 文件的路径
    log_dir = os.path.join(script_dir, '../output/logs')
    log_file_path = os.path.join(log_dir, 'training_20241123-010528.log')
    csv_file_path = log_file_path.replace('.log', '.log.csv')

    # 调用转换函数
    convert_log_to_csv(log_file_path, csv_file_path)