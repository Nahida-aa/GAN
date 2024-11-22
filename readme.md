
```sh
pip install -r requirements.txt
```
训练模型
```sh
bash scripts/run_training.sh
```
推理
```sh
bash scripts/run_inference.sh --image path_to_image.png
```
[configs/config.json](./configs/config.json)
```json
{
  "batch_size": 64,
  "learning_rate": 0.0002,
  "epochs": 100,
  "save_interval": 10,
  "path": {
    "data": "data/",
    "output": "output/",
    "weights": "output/weights/",
    "images": "output/images/",
    "logs": "output/logs/"
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
