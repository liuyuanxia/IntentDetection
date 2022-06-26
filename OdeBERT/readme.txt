代码使用环境：
Python 3.6.5
Pytorch 1.2.0
数据集：ECDT,SNIPS,FDQuestion
dastset 包含三个数据集的具体数据
figure_odebert 对模型的效果进行分析，可视化
mdoels 加载预训练的模型进行微调  
预训练语言模型下载位置：
中文：https://github.com/ymcui/Chinese-BERT-wwm
英文：https://huggingface.co/bert-base-uncased

uer BERT的基本文件

以ECDT数据集为例：
Odebert:
python run_OdeBERT.py --train_path ./datasets/ECDT/UserIntenttrain.csv --test_path ./datasets/ECDT/Usertest.csv --output_model_path ./models/ECDT/OdeBERT.bin
