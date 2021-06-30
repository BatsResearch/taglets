
def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def get_total_size(shape):
    tot = 1
    for comp in shape:
        tot += comp
    return tot
