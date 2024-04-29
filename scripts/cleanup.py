import gc

def cleanup_gpu():
    """
    Function to clean up GPU memory
    """
    # This import is in here because it's slow. Note that if called repeatedly
    # it is not expensive.
    import torch
    # Remove references to all CUDA objects
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            del obj
    gc.collect()
    torch.cuda.empty_cache()
