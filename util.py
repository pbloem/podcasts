def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

import torch, os, errno

from torch.autograd import Variable

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the file resides)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '.', subpath))

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

def makedirs(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def contains_nan(input):
    if (not isinstance(input, torch.Tensor)) and isinstance(input, Iterable):
        for i in input:
            if contains_nan(i):
                return True
        return False
    else:
        return bool(torch.isnan(input).sum() > 0)
#
def contains_inf(input):
    if (not isinstance(input, torch.Tensor)) and isinstance(input, Iterable):
        for i in input:
            if contains_inf(i):
                return True
        return False
    else:
        return bool(torch.isinf(input).sum() > 0)

def kl_loss(zmean, zsig):
    b, l = zmean.size()

    kl = 0.5 * torch.sum(zsig.exp() - zsig + zmean.pow(2) - 1, dim=1)
    # -- The KL divergence between a given normal distribution and a standard normal distribution
    #    can be rewritten this way. It's a good exercise to work this out.

    assert kl.size() == (b,)
    # -- At this point we want the loss to be a single value of each instance in the batch.
    #    Asserts like this are a good way to document what you know about the shape of the
    #    tensors you deal with.

    return kl

def sample(zmean, zsig):
    b, l = zmean.size()

    # sample epsilon from a standard normal distribution
    eps = torch.randn(b, l)

    # transform eps to a sample from the given distribution
    return zmean + eps * (zsig * 0.5).exp()
