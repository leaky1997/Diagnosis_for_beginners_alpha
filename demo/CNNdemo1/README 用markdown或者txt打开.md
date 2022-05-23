# CNN诊断代码Demo
## 阅读该代码之后需要习得一下几点：
1. 编写智能故障诊断代码的基本逻辑
    * 数据导入
    * 数据预处理
    * 模型建立
    * 模型训练（本demo，其他的请看相应的函数）
    * 模型评估与测试 （本demo包括模型评估）
2. 编写代码的基本规范
    * 注释
    * 空格
    * 变量命名
    * 其他请参考 [Google 代码风格](https://github.com/shendeguize/GooglePythonStyleGuideCN)中的第三部分
3. 本代码只提供了一个主程序的逻辑，其他相关内容请参考我的[仓库](https://gitee.com/Leaky/diagnosis_for_beginners)
    * 欢迎师弟师妹一起来贡献仓库，如果觉得麻烦可以把代码发给管理员来更新。
4. 本demo文件逻辑，程序中也有相应注释
    - sets.mat **数据集**
        - 训练集
        - 测试集
    - Demo 1 CNN.py or Demo 1 CNN.ipynb **主程序**
        - 诊断模型类
            - fit 训练
            - evaluation 评估
        - 数据读取
    - cnn_model.py **模型程序**
        - 自定义模块，例如1X1卷积
        - 利用框架构架模型，明确输入输出的size
        - 主程序随机生成input 测试模型骨架
    - readdata.py **数据预处理**
        - shuru 原始数据读取
        - norm 可以采用sklearn中的库，数据标准化
        - sampling 从序列中采样样本
        - readdata类 实现读取-数据预处理-采样的工作流，构建一个样本
        - dataset 在readdata的基础上重复执行readdate的流，来构建数据集
        - 主程序 input 数据集地址
    - utils.py **其他工具**
        - lossplot 绘制loss
        - accplot 绘制acc
        - figscat 绘制散点图
        - print_network 打印网络结构
        - initialize_weights 初始化网络参数， xavier 还是kaiming
5. 可调整参数
   - lr=0.001
   - batch_size=64
   - num_train=200
6. 损失函数
   - 本程序中主要使用了交叉熵损失函数，用于求取模型预测输出与真实标签的损失[y_pre, y]

        