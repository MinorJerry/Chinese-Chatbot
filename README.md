# 学习制作聊天机器人
## Step1:学习官方教程
- 代码和过程具体参考  https://segmentfault.com/p/1210000018096874/read  
- 使用的语料集为：小黄鸡语料 &nbsp; &nbsp; 分词工具：清华分词 
-  在xiaohuangji/文件夹下:   
>>  Linux: ```python3 train.py```  
>>  powershell: ```python .\train.py```   
- 如果要测试训练好的模型:把train.py中的loadFilename前的注释去掉,仅在loadFilename != None 时,才能利用已经训练好的模型 
- 可以通过设置checkpoint_iter, n_iteration来继续对模型进行训练,目前是500步保存一次模型,每一次保存约占1.4G
- 目前decoder是GreadySearchDecoder, 使用BeamSearchDecoder可能效果会更好(写的有点问题,到时候再看)
- 下图展示的是26000次迭代的效果,效果很大程度上取决于数据集
## Step2:Our course project
- Question-Answering System based on Knowledge Retrivaling and seq2seq.
- Part of [Code](https://github.com/ZJUGuoShuai/QA_KG)
- Document

