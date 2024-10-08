import torch
import torch.fft

def fft2(data):
    # Convert old format (real and imaginary parts separated) to complex tensor if needed
    if data.size(-1) == 2:
        data = torch.complex(data[..., 0], data[..., 1])
    data = torch.fft.fft2(data, norm="ortho")
    return torch.view_as_real(data)

def fft2c(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    if data.size(-1) == 2:
        data = torch.complex(data[..., 0], data[..., 1])
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft.fft2(data, norm="ortho")
    data = fftshift(torch.view_as_real(data), dim=(-3, -2))
    return data

def ifft2(data):
    # Convert old format (real and imaginary parts separated) to complex tensor if needed
    if data.size(-1) == 2:
        data = torch.complex(data[..., 0], data[..., 1])
    data = torch.fft.ifft2(data, norm="ortho")
    return torch.view_as_real(data)

def ifft2c(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    if data.size(-1) == 2:
        data = torch.complex(data[..., 0], data[..., 1])
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft.ifft2(data, norm="ortho")
    data = fftshift(torch.view_as_real(data), dim=(-3, -2))
    return data

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)