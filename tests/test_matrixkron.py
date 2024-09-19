from mmbo.dist import _matrixkron
import torch

torch.manual_seed(0)

a = torch.randn(3,5)
b = torch.randn(4,7)

c = _matrixkron(a, b)

assert c.shape == torch.Size([12,35])
assert (c == torch.kron(a,b)).all()

x = torch.randn(2,1,2,3,5)
y = torch.randn(1,2,2,4,7)

z = _matrixkron(x,y)

assert z.shape == torch.Size([2,2,2,12,35])

assert (z[0,0,0,:,:] == torch.kron(x[0,0,0,:,:], y[0,0,0,:,:])).all()
assert (z[0,0,1,:,:] == torch.kron(x[0,0,1,:,:], y[0,0,1,:,:])).all()
assert (z[0,1,0,:,:] == torch.kron(x[0,0,0,:,:], y[0,1,0,:,:])).all()
assert (z[0,1,1,:,:] == torch.kron(x[0,0,1,:,:], y[0,1,1,:,:])).all()
assert (z[1,0,0,:,:] == torch.kron(x[1,0,0,:,:], y[0,0,0,:,:])).all()
assert (z[1,0,1,:,:] == torch.kron(x[1,0,1,:,:], y[0,0,1,:,:])).all()
assert (z[1,1,0,:,:] == torch.kron(x[1,0,0,:,:], y[0,1,0,:,:])).all()
assert (z[1,1,1,:,:] == torch.kron(x[1,0,1,:,:], y[0,1,1,:,:])).all()

# Failure cases of torch.kron from https://github.com/pytorch/pytorch/issues/74442
_matrixkron(torch.eye(2), torch.randn(2,1))
try:
    _matrixkron(torch.eye(2), torch.randn(2,2).svd().U[:, :1]) # <- would have raised an exception
except:
    assert 1 == 0 # Will never reach here
try:
    _matrixkron(torch.eye(2), torch.randn(2,2).svd().U[:, :1].contiguous()) # <- would have raised an exception
except:
    assert 1 == 0 # Will never reach here
_matrixkron(torch.eye(2), torch.randn(2,2).svd().U.contiguous()[:, :1])