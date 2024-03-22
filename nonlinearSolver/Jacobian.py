import torch

def batchJacobian_AD_slow(y, x, graphed=False, batchx=True):
    if y.ndim == 1:
        y = y.unsqueeze(1)
    ny = y.shape[-1]
    b = y.shape[0]

    def get_vjp(v, yi):
        grads = torch.autograd.grad(outputs=yi, inputs=x, grad_outputs=v,retain_graph=True,create_graph=graphed)
        return grads


    nx = x.shape[-1]

    jacobian = torch.zeros(b,ny, nx).to(y)
    for i in range(ny):
        v = torch.ones(b).to(y)

        grad = get_vjp(v, y[:,i])[0]
        jacobian[:,i, :] = grad
    if not batchx:
        jacobian.squeeze(0)


    if not graphed:
        jacobian = jacobian.detach()

    return jacobian


