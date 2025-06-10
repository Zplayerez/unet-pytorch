# U-Net 图像语义分割项目代码说明文档

## 项目概述

本项目是基于PyTorch实现的U-Net图像语义分割框架，支持VGG16和ResNet50作为骨干网络。项目结构清晰，包含完整的数据预处理、模型训练、评估和预测功能。核心实现了U-Net架构，并提供了灵活的训练配置选项。

## 项目结构

```
.
├── .gitignore                      # Git忽略配置文件
├── LICENSE                         # MIT许可证文件
├── README.md                       # 项目说明文档
├── requirements.txt                # 项目依赖库
├── VOCdevkit/                      # VOC格式数据集目录
│   └── VOC2007/
│       ├── Convert_JPEGImages.py           # 图像格式转换脚本
│       ├── Convert_SegmentationClass.py    # 标签格式转换脚本
│       ├── ImageSets/                      # 数据集划分目录
│       │   └── Segmentation/
│       ├── JPEGImages/                     # 处理后的训练图像目录
│       ├── JPEGImages_Origin/              # 原始图像目录
│       ├── SegmentationClass/              # 处理后的标签目录
│       └── SegmentationClass_Origin/       # 原始标签目录
├── nets/                           # 网络结构定义目录
│   ├── __init__.py                        # 初始化文件
│   ├── resnet.py                          # ResNet骨干网络
│   ├── unet.py                            # U-Net网络架构
│   ├── unet_training.py                   # U-Net训练相关函数
│   └── vgg.py                             # VGG骨干网络
├── utils/                          # 工具函数目录
│   ├── __init__.py                        # 初始化文件
│   ├── callbacks.py                       # 回调函数（损失记录、评估等）
│   ├── dataloader.py                      # VOC数据集加载器
│   ├── dataloader_medical.py              # 医疗数据集加载器
│   ├── utils.py                           # 通用工具函数
│   ├── utils_fit.py                       # 训练循环函数
│   └── utils_metrics.py                   # 性能评估指标计算
├── logs/                           # 训练日志和权重保存目录
├── miou_out/                       # mIoU评估结果目录
│   └── confusion_matrix.csv               # 混淆矩阵
├── img/                            # 测试图像目录
├── img_out/                        # 预测结果输出目录
├── model_data/                     # 预训练模型和权重目录
├── Medical_Datasets/               # 医疗数据集目录（可选）
├── get_miou.py                     # 模型评估脚本
├── json_to_dataset.py              # LabelMe标注转VOC格式脚本
├── predict.py                      # 模型预测脚本
├── summary.py                      # 网络结构可视化
├── train.py                        # 模型训练脚本（标准数据集）
├── train_medical.py                # 模型训练脚本（医疗数据集）
├── unet.py                         # U-Net模型定义和推理
└── voc_annotation.py               # VOC数据集处理和划分脚本
└── voc_annotation_medical.py       # 医疗数据集处理和划分脚本
```

## 核心源码文件详解

### 1. 数据处理部分

#### `VOCdevkit/VOC2007/Convert_JPEGImages.py`

此脚本用于处理输入图像：

```python
# 关键处理逻辑
for image_name in tqdm(image_names):
    image = Image.open(os.path.join(Origin_JPEGImages_path, image_name))
    image = image.convert('RGB')  # 确保图像是RGB格式
    image.save(os.path.join(Out_JPEGImages_path, os.path.splitext(image_name)[0] + '.jpg'))
```

该脚本将任意格式的原始图像转换为RGB模式的JPG格式。

#### `VOCdevkit/VOC2007/Convert_SegmentationClass.py`

此脚本用于处理分割标签：

```python
# 核心代码
# 配置像素值转换：将255转为1（二分类）
Origin_Point_Value = np.array([0, 255])
Out_Point_Value = np.array([0, 1])

# 处理每个标签文件
for png_name in tqdm(png_names):
    png = Image.open(os.path.join(Origin_SegmentationClass_path, png_name))
    w, h = png.size
    
    png = np.array(png)
    out_png = np.zeros([h, w])
    for i in range(len(Origin_Point_Value)):
        mask = png[:, :] == Origin_Point_Value[i]
        if len(np.shape(mask)) > 2:
            mask = mask.all(-1)  # 处理RGB标签
        out_png[mask] = Out_Point_Value[i]
```

该脚本将标签图像中的像素值映射为类别索引（例如0→0, 255→1），适用于语义分割任务。

#### `voc_annotation.py`

负责创建训练集和验证集的分割，并验证数据集格式：

```python
# 数据集参数配置
trainval_percent = 1  # 训练验证集比例
train_percent = 0.9  # 训练集比例

# 核心逻辑：随机划分训练集和验证集
num = len(total_seg)  
list = range(num)  
tv = int(num*trainval_percent)  # 获取训练验证集大小
tr = int(tv*train_percent)  # 获取训练集大小
trainval = random.sample(list, tv)  # 随机采样训练验证集索引
train = random.sample(trainval, tr)  # 随机采样训练集索引
```

该脚本还会检查像素值分布，确保标签格式正确。

### 2. 模型定义部分

#### `nets/unet.py`

U-Net模型架构定义：

```python
class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone')
        
        out_filters = [64, 128, 256, 512]
        # 上采样部分
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
        # 最终输出层
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
```

实现了U-Net的经典编码器-解码器结构，支持VGG和ResNet骨干网络。

#### `nets/vgg.py` 和 `nets/resnet.py`

这两个文件分别实现了VGG16和ResNet50骨干网络。以VGG为例：

```python
def VGG16(pretrained, in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", 
                                             model_dir="./model_data")
        model.load_state_dict(state_dict)
```

支持加载预训练权重，移除了全连接层以适应分割任务。

#### `nets/unet_training.py`

实现了模型训练所需的损失函数：

```python
# 交叉熵损失
def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

# Focal Loss，用于处理类别不平衡
def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    # 实现代码...

# Dice Loss，计算预测与标签的重叠程度
def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    # 实现代码...
```

包含了交叉熵损失、Focal Loss和Dice Loss的实现，以及学习率调度器等训练工具。

### 3. 数据加载部分

#### `utils/dataloader.py`

定义了数据集加载和增强：

```python
class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        # 初始化代码...
        
    def __getitem__(self, index):
        # 加载图像和标签
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        jpg = Image.open(os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png"))
        
        # 数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)
        
        # 预处理
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes  # 处理超出类别范围的像素
        
        # One-hot编码
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        
        return jpg, png, seg_labels
```

实现了数据加载、预处理和数据增强功能，包括随机缩放、翻转、色彩变换等。

### 4. 训练流程实现

#### `train.py`

模型训练的主脚本，包含主要训练参数配置：

```python
# 关键参数设置
num_classes = 2  # 类别数+1
backbone = "vgg"  # 骨干网络
pretrained = False  # 是否使用预训练权重
model_path = "model_data/unet_vgg_voc.pth"  # 预训练模型路径
input_shape = [512, 512]  # 输入图像大小

# 训练策略参数
Freeze_Train = True  # 是否冻结训练
Init_Epoch = 0  # 起始epoch
Freeze_Epoch = 50  # 冻结训练的epoch数
UnFreeze_Epoch = 100  # 总训练epoch数
Freeze_batch_size = 2  # 冻结阶段batch size
Unfreeze_batch_size = 2  # 解冻阶段batch size
Init_lr = 1e-4  # 初始学习率
Min_lr = Init_lr * 0.01  # 最小学习率
optimizer_type = "adam"  # 优化器类型
momentum = 0.9  # 动量
weight_decay = 0  # 权重衰减
lr_decay_type = 'cos'  # 学习率衰减方式
dice_loss = False  # 是否使用Dice Loss
focal_loss = False  # 是否使用Focal Loss
```

训练流程的实现：

```python
model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
if Freeze_Train:
    model.freeze_backbone()  # 冻结骨干网络

optimizer = {
    'adam': optim.Adam(model.parameters(), Init_lr_fit, weight_decay=weight_decay),
    'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
}[optimizer_type]

# 数据加载
train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
gen = DataLoader(train_dataset, batch_size=batch_size, ...)
gen_val = DataLoader(val_dataset, batch_size=batch_size, ...)

# 训练循环
for epoch in range(Init_Epoch, UnFreeze_Epoch):
    # 学习率调整、模型解冻等逻辑
    fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
              epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, 
              focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)
```

#### `utils/utils_fit.py`

实现了单个epoch的训练和验证逻辑：

```python
def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    # 训练循环
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        
        # 计算损失
        outputs = model_train(imgs)
        if focal_loss:
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
        else:
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice
            
        # 反向传播
        loss.backward()
        optimizer.step()
        
    # 验证循环 
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        # 验证逻辑...
        
    # 保存模型
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))
```

### 5. 评估和预测部分

#### `get_miou.py`

用于评估模型在验证集上的性能：

```python
# 关键参数
miou_mode = 0  # 评估模式
num_classes = 2  # 类别数+1

# 评估流程
if miou_mode == 0 or miou_mode == 1:
    # 获取预测结果
    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image = Image.open(image_path)
        image = unet.get_miou_png(image)  # 获取预测分割图
        image.save(os.path.join(pred_dir, image_id + ".png"))

if miou_mode == 0 or miou_mode == 2:
    # 计算mIoU
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
    # 显示结果
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
```

#### `predict.py`

用于使用训练好的模型进行预测：

```python
# 预测模式选择
mode = "dir_predict"  # predict/video/fps/dir_predict/export_onnx/predict_onnx

# 参数设置
count = False  # 是否计算像素数量
name_classes = ["background", "aeroplane", ...]  # 类别名称

# 模型配置
unet = Unet()  # 创建模型实例

# 预测函数示例（目录预测模式）
if mode == "dir_predict":
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)
            r_image = unet.detect_image(image)  # 进行预测
            r_image.save(os.path.join(dir_save_path, img_name))  # 保存结果
```

#### `unet.py`

模型推理的实现：

```python
def detect_image(self, image, count=False, name_classes=None):
    # 图像预处理
    image = cvtColor(image)  # 转RGB
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    
    # 模型预测
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()
        pr = self.net(images)[0]  # 前向传播获取预测
        pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy()
        
        # 后处理
        pr = pr[int((self.input_shape[0] - nh) // 2):int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2):int((self.input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)  # 获取每个像素的类别
    
    # 可视化
    if self.mix_type == 0:
        # 混合可视化模式
        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))
        image = Image.blend(old_img, image, 0.7)  # 原图与分割结果混合
```

## 模型参数详解

### 1. U-Net模型参数

- **num_classes**: 分类类别总数（包括背景），例如二分类设为2
- **backbone**: 骨干网络类型，默认"vgg"，也支持"resnet50"
- **pretrained**: 是否使用预训练权重，默认False

### 2. 训练相关参数

- **Freeze_Train**: 是否进行冻结训练，默认True，冻结骨干网络进行特征提取
- **Init_Epoch/Freeze_Epoch/UnFreeze_Epoch**: 初始轮次/冻结训练轮次/总训练轮次
- **Freeze_batch_size/Unfreeze_batch_size**: 冻结/解冻阶段批大小
- **Init_lr/Min_lr**: 初始学习率/最小学习率，默认1e-4/1e-6
- **optimizer_type**: 优化器类型，"adam"或"sgd"
- **momentum**: SGD优化器的动量参数，默认0.9
- **weight_decay**: 权重衰减，防止过拟合，adam优化器建议设为0
- **lr_decay_type**: 学习率衰减方式，"cos"或"step"
- **dice_loss/focal_loss**: 是否使用Dice Loss和Focal Loss，用于处理类别不平衡问题

### 3. 数据相关参数

- **input_shape**: 输入图片大小，建议为32的倍数，例如[512, 512]
- **VOCdevkit_path**: 数据集路径，默认"VOCdevkit"
- **trainval_percent/train_percent**: 训练验证集比例/训练集比例，默认1.0/0.9

## 模型执行流程

### 1. 数据预处理流程

1. **原始数据准备**:
   - 将原始图像放入`JPEGImages_Origin`目录
   - 将分割标签放入`SegmentationClass_Origin`目录
2. **图像格式转换**:
   - 执行`Convert_JPEGImages.py`，将图像转为RGB的JPG格式
   - 执行`Convert_SegmentationClass.py`，将标签转为单通道PNG格式，像素值映射为类别索引
3. **数据集划分**:
   - 执行`voc_annotation.py`，随机划分训练集和验证集
   - 生成`train.txt`和`val.txt`，记录对应图像的基础名称
   - 检查标签像素值分布，确保数据格式正确

### 2. 模型训练流程

1. **模型初始化**:
   - 根据`backbone`参数选择骨干网络(VGG16或ResNet50)
   - 加载预训练权重(如果`pretrained=True`)
   - 构建U-Net编码器-解码器结构
2. **冻结训练阶段** (如果`Freeze_Train=True`):
   - 冻结骨干网络权重，只训练解码器部分
   - 使用`Freeze_batch_size`设置批大小
   - 训练`Freeze_Epoch`轮
3. **解冻训练阶段**:
   - 解冻骨干网络，整体微调
   - 使用`Unfreeze_batch_size`设置批大小
   - 从`Freeze_Epoch`训练到`UnFreeze_Epoch`轮
4. **训练细节**:
   - 每轮训练后在验证集上评估性能
   - 根据`save_period`定期保存模型权重
   - 自动保存验证集性能最好的模型为`best_epoch_weights.pth`

### 3. 模型评估流程

1. **模型加载**:
   - 加载训练好的权重文件
   - 构建相同架构的U-Net模型
2. **预测验证集**:
   - 对验证集中的每张图像进行预测
   - 生成预测的分割图并保存
3. **计算评估指标**:
   - 计算每个类别的IoU(交并比)
   - 计算mIoU(平均交并比)
   - 计算Precision(精确率)和Recall(召回率)
   - 生成混淆矩阵和可视化结果

### 4. 模型预测流程

1. **模型加载**:
   - 加载训练好的模型和权重
   - 配置推理参数(如混合类型、颜色映射等)
2. **输入预处理**:
   - 读取输入图像并转为RGB格式
   - 调整图像大小并进行归一化
3. **模型推理**:
   - 执行前向传播得到预测结果
   - 对输出进行softmax处理获取每个像素的类别概率
   - 选择概率最高的类别作为最终分割结果
4. **结果后处理**:
   - 将分割结果调整回原始图像大小
   - 根据`mix_type`参数可视化结果(原图混合/纯分割图/只显示前景)
   - 保存结果图像

通过以上流程，可以完成从数据准备到模型训练、评估和预测的全过程。
