from timm.models import create_model
import numpy as np
import torch


def cal_Acc(confusion):
    """
    计算准确率

    :param confusion: 混淆矩阵
    :return: Acc
    """
    return np.sum(confusion.diagonal())/np.sum(confusion)


def cal_Pc(confusion):
    """
    计算每类精确率

    :param confusion: 混淆矩阵
    :return: Pc
    """
    return confusion.diagonal()/np.sum(confusion,axis=1)


def cal_Rc(confusion):
    """
    计算每类召回率

    :param confusion: 混淆矩阵
    :return: Rc
    """
    return confusion.diagonal()/np.sum(confusion,axis=0)


def cal_F1score(PC, RC):
    """
    计算F1 score

    :param PC: 精准率
    :param RC: 召回率
    :return: F1 score
    """
    return 2*np.multiply(PC,RC)/(PC+RC)


# 加载预训练模型的实现函数
def load_pth_model(mymodel, pth_path, num):
    my_model = create_model(
        mymodel,
        pretrained=False,
        num_classes=num,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )

    checkpoint = torch.load(pth_path)
    my_model.load_state_dict(checkpoint)
    my_model.eval()

    return my_model




