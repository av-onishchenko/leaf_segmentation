import numpy as np
from scipy.optimize import linear_sum_assignment


def DiffFgDICE(target, predicted):
    TP = np.sum((target > 0.5) * (predicted > 0.5), axis=(-2, -1))
    FP = np.sum((target <= 0.5) * (predicted > 0.5), axis=(-2, -1))
    FN = np.sum((target > 0.5) * (predicted <= 0.5), axis=(-2, -1))
    iou = TP / (TP + FP + FN + 1e-10)
    return np.sum(2 * iou / (1 + iou))


def DiffFgMSE(target, predicted):
    num_of_target_pixels = np.sum(target)
    return np.sum((target - predicted) ** 2 * target) / num_of_target_pixels


def AbsDiffFGLabels(target, predicted):
    return np.absolute(target - np.round(predicted))


def DiffFGLabels(target, predicted):
    return target - np.round(predicted)


def SymmetricBestDice(target, predicted, num_target, num_predicted):
    dice_tables = np.zeros((num_target, num_predicted))
    for i in range(num_target):
        for j in range(num_predicted):
            cur_target = (target == i + 1).reshape(1, target.shape[0], target.shape[1])
            cur_predicted = (predicted == j + 1).reshape(
                1, target.shape[0], target.shape[1]
            )
            dice_tables[i][j] = DiffFgDICE(cur_target, cur_predicted)
    dice_tables = 1 - dice_tables
    row_ind, col_ind = linear_sum_assignment(dice_tables)
    return (1 - dice_tables[row_ind, col_ind]).mean()


def SemanticSegMetrics(target, predicted):
    return DiffFgDICE(target, predicted), DiffFgMSE(target, predicted)


def InstanceSegMetrics(target, predicted):
    return DiffFGLabels(target, predicted), AbsDiffFGLabels(target, predicted)
