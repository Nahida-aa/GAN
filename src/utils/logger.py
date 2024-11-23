# src/utils/logger.py
import logging
import os

def get_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('"%(asctime)s",%(message)s')
    file_handler = logging.FileHandler(log_file)
    # 检查文件是否存在，如果不存在则写入表头
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('"Timestamp","Epoch","Batch","D Loss","G Loss"\n')
    file_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    return logger