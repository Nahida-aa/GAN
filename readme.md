## overview
一个基于PyTorch的GAN实现，用于生成手写数字图像。

- [data](./data/): 默认数据文件夹
- [docs](./docs/): 文档文件夹
- [notebooks](./notebooks/): Jupyter notebooks文件夹
- [src](./src/): 源代码文件夹
  - [data](./src/data/): 数据加载和处理文件夹
  - [models](./src/models/): 模型定义文件夹
  - [training](./src/training/): 训练文件夹
  - [inference](./src/inference/): 推理文件夹
  - [utils](./src/utils/): 工具文件夹
- [tests](./tests/): 测试文件夹
- [configs](./configs/): 配置文件夹
- [scripts](./scripts/): 脚本文件夹
- [output](./output/): 输出文件夹
  - [weights](./weights/): 模型权重文件夹
  - [images](./images/): 生成的图像文件夹
  - [logs](./logs/): 日志文件夹

## features
1. **数据处理**：包括数据加载、预处理、增强等。
2. **模型定义**：定义模型的架构和超参数。
3. **训练**：包括训练循环、损失计算、优化器设置等。
4. **验证和测试**：包括验证集和测试集的评估。
5. **模型保存和加载**：保存训练好的模型，并能够加载进行推理。
6. **推理**：使用训练好的模型进行预测。
7. **日志记录和可视化**：记录训练过程中的日志，并可视化训练曲线。
8. **配置管理**：使用配置文件管理超参数和路径等。
9. **单元测试**：对关键功能进行单元测试，确保代码的正确性。
10. **文档**：详细的README文件，包含项目介绍、安装步骤、使用方法等。

## using
### install dependencies
```sh
pip install -r requirements.txt
```
### 训练模型
```sh
python src/training/train.py
```
### 推理
```sh
# 现在还没有好的训练结果，所以暂时没写推理脚本
```

## config
[configs/config.json](./configs/config.json)
```json
{
  "training_parameters": {
    "batch_size": 64,
    "learning_rate": 0.0002,
    "epochs": 1000,
    "save_intervals": [
      {"start": 0, "end": 10, "interval": 1},
      {"start": 10, "end": 50, "interval": 5},
      {"start": 50, "end": 1000, "interval": 10}
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
