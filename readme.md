
```sh
pip install -r requirements.txt
```
训练模型
```sh
python src/training/train.py
```
推理
```sh
# 现在还没有好的训练结果，所以暂时没写推理脚本
```
[configs/config.json](./configs/config.json)
```json
{
  "training_parameters": {
    "batch_size": 64,
    "learning_rate": 0.0002,
    "epochs": 200,
    "save_intervals": [
      {"start": 0, "end": 10, "interval": 1},
      {"start": 10, "end": 50, "interval": 5},
      {"start": 50, "end": 200, "interval": 10}
    ]
  },
  "path": {
    "data_dir": "data/",
    "output_dir": "output/",
    "weights_dir": "output/weights/",
    "images_dir": "output/images/",
    "logs_dir": "output/logs/"
  }
}
```

- [data](./data/): 默认数据文件夹
- [notebooks](./notebooks/): Jupyter notebooks文件夹
- [src](./src/): 源代码文件夹
- [tests](./tests/): 测试文件夹
- [configs](./configs/): 配置文件夹
- [scripts](./scripts/): 脚本文件夹
- [output](./output/): 输出文件夹
  - [weights](./weights/): 模型权重文件夹
  - [images](./images/): 生成的图像文件夹
  - [logs](./logs/): 日志文件夹
