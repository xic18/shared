# Trainer

https://zhuanlan.zhihu.com/p/363670628

https://zhuanlan.zhihu.com/p/662619853

基本参数：

> *class*`transformers.Trainer`(
>
> ***model**: torch.nn.modules.module.Module = None*,
>
> ***args**: transformers.training_args.TrainingArguments = None*,
>
> ***data_collator**: Optional[NewType.<locals>.new_type] = None*,
>
> ***train_dataset**: Optional[torch.utils.data.dataset.Dataset] = None*,
>
> ***eval_dataset**: Optional[torch.utils.data.dataset.Dataset] = None*,
>
> ***tokenizer**: Optional[transformers.tokenization_utils_base.PreTrainedTokenizerBase] = None*,
>
> ***model_init**: Callable[transformers.modeling_utils.PreTrainedModel] = None*,
>
> ***compute_metrics**: Optional[Callable[transformers.trainer_utils.EvalPrediction*,*Dict]] = None*,
>
> ***callbacks**: Optional[List[transformers.trainer_callback.TrainerCallback]] = None*,
>
> ***optimizers**: Tuple[[torch.optim.optimizer.Optimizer](https://zhida.zhihu.com/search?content_id=168944533&content_type=Article&match_order=1&q=torch.optim.optimizer.Optimizer&zhida_source=entity)*,*torch.optim.lr_scheduler.LambdaLR] = (None*,*None)*)



## TrainingArgument

- **output_dir** (`str`) – 我们的模型训练过程中可能产生的文件存放的路径，包括了模型文件，checkpoint，log文件等；



**evaluation_strategy** 

用steps比较方便，因为可以通过后面的eval steps来控制eval的频率，每个epoch的话感觉太费时间了



**gradient_accumulation_steps**

显存重计算的技巧，很方便很实用，默认为1，如果设置为n，则我们forward n次，得到n个loss的累加后再更新参数。

显存重计算是典型的用时间换空间，比如我们希望跑256的大点的batch，不希望跑32这样的小batch，因为觉得小batch不稳定，会影响模型效果，但是gpu显存又无法放下256的batchsize的数据，此时我们就可以进行显存重计算，将这个参数设置为256/32=8即可。用torch实现就是forward，计算loss 8次，然后再optimizer.step()
注意，当我们设置了显存重计算的功能，则eval  steps之类的参数自动进行相应的调整，比如我们设置这个参数前，256的batch，我们希望10个batch评估一次，即10个steps进行一次eval，当时改为batch size=32并且 gradient_accumulation_steps=8，则默认trainer会 8*10=80个steps  进行一次eval。













adam怎么考虑L1 L2 正则（有待研究，目前的感觉是nlp里面预训练模型的知识都是存储在weights里面的，所以不太适合做L1正则，也不太做layernorm）

- **L2 正则**：等价于权重衰减（Weight Decay），通过向损失函数添加参数的平方和惩罚项（`λ||θ||²`）。
- **L1 正则**：通过向损失函数添加参数的绝对值惩罚项（`λ||θ||`），常用于稀疏化模型参数。
- **关键区别**：
  - L2 正则通过梯度下降直接作用于参数更新（如 SGD 的 `θ = θ - η(∇L + λθ)`）。
  - **Adam 的权重衰减实现不同**：标准的 Adam 会将权重衰减混入梯度计算（可能导致不理想的正则化效果），标准 Adam 的 `weight_decay` 实际是 **逐参数缩放**，而非真正的 L2 正则。若需严格 L2 效果，应使用 **AdamW**。 **AdamW** 解耦了权重衰减，更接近真正的 L2 正则。







trainer怎么只使用单卡训练

os.environ['CUDA_VISIBLE_DEVICES']='0' 即可（必须在 `CUDA_VISIBLE_DEVICES` 之前设置 `CUDA_DEVICE_ORDER`，否则可能不生效）

```python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  

import torch
print(torch.cuda.device_count())  # 输出 2（可见 GPU 数量）
```

`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` 是一个环境变量设置，用于 **强制禁用 GPU**，使程序只能使用 CPU 进行计算

其他环境变量：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定可见的 GPU
WORLD_SIZE=4                   # 总进程数（通常等于 GPU 数）
RANK=0,1,2,3                   # 每个进程的唯一编号
MASTER_ADDR="localhost"        # 主节点地址（多机时需设置）
MASTER_PORT=12345              # 主节点端口
```



并行训练有用的教程：

https://huggingface.co/blog/zh/pytorch-ddp-accelerate-transformers







# early stop

### **1. 使用 `EarlyStoppingCallback`（推荐）**

Transformers 库内置了早停回调，可直接使用：

#### **(1) 导入并配置回调**

python

```
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

# 定义早停回调（监控验证集损失，patience=3 表示连续3次指标未提升则停止）
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # 容忍的停滞轮次/步数
    early_stopping_threshold=0.001,  # 最小提升阈值（可选）
)

# 在 TrainingArguments 中启用验证评估
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",  # 按步评估（可选 "epoch"）
    eval_steps=500,              # 每500步评估一次
    save_strategy="steps",       # 保存策略与评估一致
    load_best_model_at_end=True, # 训练结束时加载最佳模型
    metric_for_best_model="eval_loss",  # 监控的指标（如 eval_loss、eval_accuracy）
    greater_is_better=False,     # eval_loss 越小越好（准确率则设为 True）
)

# 初始化 Trainer 并添加回调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[early_stopping],  # 关键：添加早停回调
    compute_metrics=compute_metrics,  # 自定义指标计算函数（可选）
)
trainer.train()
```

#### **(2) 参数说明**

| 参数名                     | 作用                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `early_stopping_patience`  | 允许指标未提升的最大连续评估次数（如设为3，则第4次未提升时停止）。 |
| `early_stopping_threshold` | 指标需提升的最小绝对值（如 `0.001` 表示提升需超过此值才不算停滞）。 |
| `metric_for_best_model`    | 监控的指标名（需与 `compute_metrics` 返回的字典键名一致）。  |
| `greater_is_better`        | 指标是否越大越好（如准确率为 `True`，损失为 `False`）。      |

------

### **2. 自定义早停逻辑（高级需求）**

如果内置回调不满足需求（如需监控多个指标），可继承 `TrainerCallback` 实现自定义逻辑：

#### **(1) 自定义回调示例**

python

```
from transformers import TrainerCallback

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get("eval_loss")  # 监控 eval_loss
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric > self.best_metric - self.min_delta:  # 未提升
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True  # 触发早停
        else:  # 有提升
            self.best_metric = current_metric
            self.counter = 0

# 使用时替换内置回调
trainer = Trainer(..., callbacks=[CustomEarlyStoppingCallback(patience=3)])
```

#### **(2) 关键方法**

- `on_evaluate`：在每次评估后触发，可访问评估指标 `metrics`。
- `control.should_training_stop`：设为 `True` 以终止训练。

------

### **3. 结合模型保存**

为确保早停时保存最佳模型，需设置：

python

```
training_args = TrainingArguments(
    ...
    save_strategy="steps",             # 与 evaluation_strategy 一致
    save_steps=500,                   # 与 eval_steps 一致
    load_best_model_at_end=True,      # 训练结束加载最佳模型
    metric_for_best_model="eval_loss", # 监控的指标
    greater_is_better=False,          # 越小越好
)
```

------

### **4. 完整流程示例**

python

```
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# 加载数据和模型
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 定义评估指标（早停依据）
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    accuracy = (predictions.argmax(-1) == labels).mean()
    return {"eval_accuracy": accuracy}

# 配置早停回调
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.01,
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,  # 准确率越大越好
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# 开始训练（会自动早停）
trainer.train()
```

------

### **5. 注意事项**

1. **评估频率**：
   - `evaluation_strategy` 和 `eval_steps` 需合理设置，避免评估过于频繁拖慢训练。
2. **指标一致性**：
   `metric_for_best_model` 必须与 `compute_metrics` 返回的字典键名完全一致。
3. **多指标监控**：
   内置回调仅支持单指标，如需多指标需自定义回调。
4. **分布式训练**：
   早停决策会自动同步到所有进程，无需手动处理。

------

### **总结**

- **推荐方法**：直接使用 `EarlyStoppingCallback`，配置 `patience` 和监控指标。
- **高级需求**：继承 `TrainerCallback` 实现自定义逻辑。
- **确保生效**：设置 `load_best_model_at_end=True` 以保存最佳模型。







# 输出

在使用 Hugging Face Transformers 的 `Trainer` 进行训练后，生成的 **`mlruns`** 和 **`output_dir`**（即你在 `TrainingArguments` 中指定的 `output_dir`，如 `"./results"`）是两个功能不同的文件夹，分别用于 **实验跟踪** 和 **模型/训练结果存储**。以下是它们的详细解释：

------

### **1. `output_dir`（训练输出目录）**

这是通过 `TrainingArguments(output_dir="your_path")` 指定的文件夹，存放与训练直接相关的文件。
**典型结构**：

bash

```
output_dir/
├── checkpoint-500/          # 保存的检查点（每N步或epoch保存一次）
│   ├── pytorch_model.bin    # 模型权重
│   ├── optimizer.pt         # 优化器状态
│   ├── scheduler.pt         # 学习率调度器状态
│   ├── trainer_state.json   # 训练状态（步数、epoch、最佳指标等）
│   └── config.json          # 模型配置文件
├── checkpoint-1000/
├── ...                      
├── logs/                    # 训练日志（可选，需设置 logging_dir）
│   └── events.out.tfevents...  # TensorBoard 日志
└── best_model/              # 最佳模型（若设置 load_best_model_at_end=True）
```

#### **关键文件作用**：

| 文件/目录            | 用途                                                         |
| -------------------- | ------------------------------------------------------------ |
| `checkpoint-*/`      | 训练过程中保存的检查点（模型、优化器、调度器状态），可用于恢复训练。 |
| `pytorch_model.bin`  | PyTorch 模型权重文件。                                       |
| `trainer_state.json` | 训练进度（如当前 step、epoch、最佳指标值）。                 |
| `config.json`        | 模型结构配置（如 BERT 的层数、隐藏层大小等）。               |
| `best_model/`        | 训练结束时保存的最佳模型（需设置 `load_best_model_at_end=True`）。 |
| `logs/`              | TensorBoard 日志（需设置 `logging_dir`）。                   |

------

### **2. `mlruns`（MLflow 实验跟踪目录）**

如果安装了 `mlflow` 库（或使用了 `--report_to mlflow` 参数），`Trainer` 会自动生成此文件夹，用于 **记录实验参数、指标和模型**（类似 TensorBoard，但更全面）。
**典型结构**：

bash

```
mlruns/
├── 0/                      # 实验ID
│   ├── meta.yaml           # 实验元数据
│   └── [run_id]/           # 每次运行的唯一ID
│       ├── metrics/        # 记录的指标（如 eval_loss、accuracy）
│       ├── params/         # 超参数（如 learning_rate、batch_size）
│       ├── artifacts/      # 保存的模型/文件（可选）
│       └── tags/           # 实验标签
└── ...
```

#### **关键文件作用**：

| 文件/目录    | 用途                                                         |
| ------------ | ------------------------------------------------------------ |
| `metrics/`   | 训练/验证指标（如每步的 loss、accuracy），可用于可视化。     |
| `params/`    | 训练超参数（如 `per_device_batch_size`、`num_train_epochs`）。 |
| `artifacts/` | 保存的模型或文件（需主动调用 `mlflow.log_artifact()`）。     |
| `tags/`      | 实验标签（如任务名称、模型类型）。                           |

------

### **3. 两者的核心区别**

| 特性         | `output_dir`                       | `mlruns`                                      |
| ------------ | ---------------------------------- | --------------------------------------------- |
| **目的**     | 存储模型、检查点、训练状态。       | 记录实验参数、指标、元数据（用于分析）。      |
| **生成条件** | 始终生成（由 `output_dir` 指定）。 | 需安装 `mlflow` 或设置 `--report_to mlflow`。 |
| **内容**     | 模型权重、优化器状态等二进制文件。 | 结构化指标和参数（文本/JSON 格式）。          |
| **工具依赖** | 无需额外工具。                     | 需 MLflow 或 TensorBoard 可视化。             |

------

### **4. 如何控制它们的生成？**

#### **(1) 禁用 `mlruns`**

如果不想要 MLflow 记录，可通过以下方式禁用：

python

```
training_args = TrainingArguments(
    output_dir="./results",
    report_to="none",  # 关闭所有日志记录（包括 MLflow、TensorBoard）
)
```

#### **(2) 自定义 `output_dir` 内容**

通过 `TrainingArguments` 控制检查点保存频率：

python

```
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="steps",      # 按步保存（可选 "epoch"）
    save_steps=500,             # 每 500 步保存一次
    save_total_limit=2,         # 最多保留 2 个检查点（自动删除旧的）
)
```

------

### **5. 实际使用建议**

- **调试时**：关注 `output_dir/checkpoint-*/` 下的模型和 `trainer_state.json`。

- **实验分析时**：使用 `mlflow ui` 启动本地服务器查看指标趋势：

  bash

- ```
  mlflow ui --port 5000  # 浏览器访问 http://localhost:5000
  ```

- **部署模型时**：直接使用 `output_dir/best_model/` 或最新检查点。

------

### **总结**

- **`output_dir`**：核心训练输出，包含模型、检查点和日志，必须保留。
- **`mlruns`**：实验跟踪数据，适合超参数分析和实验对比，可选择性启用。
- **优先级**：若磁盘空间不足，可定期清理 `mlruns`，但务必保留 `output_dir`。

如果使用tensorboard？

要查看使用 `logging_dir="./logs"` 和 `report_to="tensorboard"` 配置的 Transformer 训练结果，你需要使用 **TensorBoard** 工具来可视化日志文件。以下是详细步骤：

------

### **1. 确保 TensorBoard 已安装**

如果尚未安装 TensorBoard，通过 pip 安装：

bash

```
pip install tensorboard
```

------

### **2. 启动 TensorBoard**

在终端运行以下命令，指定日志目录（即 `./logs`）：

bash

```
tensorboard --logdir ./logs
```

默认情况下，TensorBoard 会在 `http://localhost:6006` 启动一个本地服务。

------

### **3. 访问 TensorBoard 界面**

打开浏览器，访问 TensorBoard 的地址：

text

```
http://localhost:6006
```

你会看到如下内容：

- **训练指标**（如损失、准确率、学习率等）。
- **直方图**（如权重分布）。
- **计算图**（模型结构，如果已记录）。
- **其他自定义标量**。

------

### **4. 关键目录结构验证**

确保你的日志目录结构正确。训练后，`./logs` 下应包含类似如下的文件：

text

```
./logs
├── events.out.tfevents.1234567890.hostname
└── (其他 TensorBoard 日志文件)
```

如果目录为空，请检查训练代码是否成功写入了日志。

------

### **5. 常见问题排查**

- **端口冲突**：如果默认端口 `6006` 被占用，改用其他端口：

  bash

- ```
  tensorboard --logdir ./logs --port 6007
  ```

- **日志路径错误**：确保 `--logdir` 指向的路径与代码中的 `logging_dir` 完全一致（建议使用绝对路径）。

- **无数据**：检查训练代码是否调用了 `trainer.train()` 并正常完成。

------

### **6. 直接查看日志文件（可选）**

如果想直接解析日志文件（如调试），可以使用 `tensorboard.summary` 或第三方库 `tbparse`：

python

```
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator("./logs/events.out.tfevents.xxx")
ea.Reload()
print(ea.Tags())  # 查看所有记录的指标
print(ea.Scalars("loss"))  # 获取损失值列表
```

在notebook使用tb

https://tensorflow.google.cn/tensorboard/tensorboard_in_notebooks?hl=zh-cn





如何设置log内容？





如何设置早停

回调函数是训练过程中的钩子，它们允许开发者在训练的特定阶段插入自定义逻辑。常见的回调类型包括：

    OnEpochBegin/OnEpochEnd：在每个epoch开始和结束时调用。
    OnBatchBegin/OnBatchEnd：在每个batch处理开始和结束时调用。
    OnTrainBegin/OnTrainEnd：在整个训练开始和结束时调用。
    ModelCheckpoint：在训练过程中定期保存模型的状态，以便可以在中断后恢复训练。
    EarlyStopping：监控模型的性能，如果在一定数量的epoch后性能没有改善，则停止训练。
    ReduceLROnPlateau：当模型的性能在一定数量的epoch后停止提升时，减少学习率。
    TensorBoard：记录训练过程中的各种指标，以便在TensorBoard中可视化。


https://huggingface.co/docs/transformers/zh/main_classes/callback#transformers.TrainerState

https://hugging-face.cn/docs/transformers/main_classes/callback

自定义step

https://blog.csdn.net/qq_38642635/article/details/118802935

```python
class PrintGradientTrainer(Trainer):
 
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
 
        loss = self.compute_loss(model, inputs)
 
        loss.backward()
        
        # ------------------------new added codes.--------------------------
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print("{}, gradient: {}".format(name, param.grad.mean()))
                else:
                    print("{} has not gradient".format(name))
        # ------------------------new added codes.--------------------------
        return loss.detach()
 
# originally the Trainer() is called
#trainer = Trainer(
#    model=model, args=training_args, train_dataset=small_train_dataset, #eval_dataset=small_eval_dataset,
#    tokenizer=tokenizer, data_collator=data_collator
#)
 
# Now call the new defined PrintGradientTrainer()
trainer = PrintGradientTrainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset,
    tokenizer=tokenizer, data_collator=data_collator
)
 
trainer.train()

```

https://blog.csdn.net/kljyrx/article/details/140153379


### model init issue
https://discuss.huggingface.co/t/can-trainer-hyperparameter-search-also-tune-the-drop-out-rate/5455/2



trainer源码解读

https://blog.csdn.net/weixin_38252409/article/details/139168463?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-2-139168463-blog-118802935.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-2-139168463-blog-118802935.235%5Ev43%5Econtrol&utm_relevant_index=4
