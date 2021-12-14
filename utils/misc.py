import time
import wandb


#@TODO use logx instead of logger!

def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        wandb.log({"Timer":"   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
                                        (f.__name__, seconds, seconds / 60, seconds / 3600)})
        return result

    return timed


def print_cuda_statistics():
    logger = ""
    import sys
    from subprocess import call
    import torch
    logger += '__Python VERSION:  {}'.format(sys.version)
    logger += '__pyTorch VERSION:  {}'.format(torch.__version__)
    logger += '__CUDA VERSION'
    call(["nvcc", "--version"])
    logger += '__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version())
    logger += '__Number CUDA Devices:  {}'.format(torch.cuda.device_count())
    logger += '__Devices'
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger += 'Active CUDA Device: GPU {}'.format(torch.cuda.current_device())
    logger += 'Available devices  {}'.format(torch.cuda.device_count())
    logger += 'Current cuda device  {}'.format(torch.cuda.current_device())
    print({"Cuda statistics ":logger})