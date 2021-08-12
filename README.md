# singing
基于pytorch的情感分析
====================
    首先是对online_shopping_10_cats进行数据探索性分析以及处理，见data_process.py，得到训练集测试集验证集
    其次在建立完模型后，需要对选用的模型textcnn或者transformer进行训练，从而获得我们的saved_models。
    注意的是：在选择textcnn或者transformer其中一个训练时，在run.py里的Edit Configurations设置Paramertes
    之后在Predict.py进行句子情感预测。
