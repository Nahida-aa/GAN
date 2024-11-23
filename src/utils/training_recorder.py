import os
import json
import datetime
from src.config import LOGS_DIR, TRAINING_PARAMETERS

def create_training_record():
    # 创建带有时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"training_{timestamp}.log.csv")

    # 记录训练参数和日志文件名到 JSON 文件
    training_record = {
        "timestamp": timestamp,
        "log_file": log_file,
        "training_parameters": TRAINING_PARAMETERS
    }
    record_file = os.path.join(LOGS_DIR, 'training_records.json')
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            records = json.load(f)
    else:
        records = []

    records.append(training_record)
    with open(record_file, 'w') as f:
        json.dump(records, f, indent=4)

    return log_file