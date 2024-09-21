import numpy as np
from scipy import stats
from argparse import ArgumentParser

# compute pooled standard error 
def pooled_standard_error(n1, n2, se1, se2):
    return np.sqrt(((n1-1)*se1**2 + (n2-1)*se2**2) / (n1 + n2 - 2))

# compute z-score
def z_score(mean1, mean2, pooled_se, n1, n2):
    return (mean1 - mean2) / pooled_se / np.sqrt(1/n1 + 1/n2)

# compute p-value
def p_value(z_score):
    return 2 * (1 - stats.norm.cdf(abs(z_score)))

n1=200
n2=200
se1 = 3.5
se2 = 3.5



x2 = 56 
x1 = 50 

# from: https://www.sciencedirect.com/science/article/pii/S0741521402000307 
# x2-x1 = 1.96*se1 +1.96*se2 - 2* 1.96 * overlap * pooled_se

# two-sample z test for independent sample
# 1.96 = 2**0.5 * 1.96 * (1-p)
# (1-p) * 2**0.5 > 1 
# 1-p > 1 / (2**0.5)

def get_target_mean_diff(n1, n2, se1, se2): 
    p = 1 - 1 / (2**0.5)

    pooled_se = pooled_standard_error(n1, n2, se1, se2)

    window = 1.96 * pooled_se * 2 
    actual_overlap = x1 + 1.96*se1 - (x2 - 1.96*se2 )

    significant = actual_overlap / window < p 

    # print(window)
    # print(actual_overlap)
    # print(p)
    max_allowed_overlap = window * p
    # print(f"{actual_overlap:.2f} vs. {max_allowed_overlap:.2f}")
    # print(significant)

    # minimum mean diff that gets statistical significance
    target_mean_diff = 1.96*se1 + 1.96*se2 - max_allowed_overlap

    print(target_mean_diff)

    return target_mean_diff

# target_mean_diff = get_target_mean_diff(400, 400, 2, 2)
# target_mean_diff = get_target_mean_diff(20,20, 11, 11)

import sys 

if __name__ == "__main__":
    args = sys.argv[1:]

    n1 = int(args[0])
    n2 = int(args[1])
    se1 = float(args[2])
    se2 = float(args[3])


    target_mean_diff = get_target_mean_diff(n1, n2, se1, se2)