def get_total_size(shape):
    tot = 1
    for comp in shape:
        tot += comp
    return tot

def mem_usage():
    res = torch.cuda.memory_reserved(0) 
    allc = torch.cuda.memory_allocated(0)
    total = torch.cuda.get_device_properties(0).total_memory

    free = res - allc  # free inside reserved
    return f'gpu mem usage: {fre} out of {t}'