def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg) 
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")

import torch 
def cross_entropy(y,y_pre,weight):
    res = weight * (y*torch.log(y_pre))
    loss=-torch.sum(res)
    return loss/y_pre.shape[0]

