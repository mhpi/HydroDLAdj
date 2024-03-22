import torch
import sourcedefender
from HydroDLAdj.nonlinearSolver.BatchJacobian_AD import batchJacobian_AD
from HydroDLAdj.nonlinearSolver.Jacobian import batchJacobian_AD_slow
#import pydevd
matrixSolve = torch.linalg.solve
class NewtonSolve(torch.autograd.Function):

  @staticmethod
  def forward(ctx, p, p2, t,G, x0=None, auxG=None, batchP=True,eval = False,AD_efficient = True):


    useAD_jac=True
    if x0 is None and p2 is not None:
      x0 = p2

    x = x0.clone().detach(); i=0;
    max_iter=3; gtol=1e-3;

    if useAD_jac:
        torch.set_grad_enabled(True)

    x.requires_grad = True

    if p2 is None:
        gg = G(x, p, t, auxG)
    else:
        gg = G(x, p, p2, t, auxG)
    if AD_efficient:
        dGdx = batchJacobian_AD(gg,x,graphed=True)
    else:
        dGdx = batchJacobian_AD_slow(gg, x, graphed=True)
    if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
        raise RuntimeError(f"Jacobian matrix is NaN")
    x = x.detach()

    torch.set_grad_enabled(False)
    resnorm = torch.linalg.norm(gg, float('inf'),dim= [1]) # calculate norm of the residuals
    resnorm0 = 100*resnorm;


    while ((torch.max(resnorm)>gtol ) and  i<=max_iter):
        i+=1
        if torch.max(resnorm/resnorm0) > 0.2:
              if useAD_jac:
                torch.set_grad_enabled(True)

              x.requires_grad = True

              if p2 is None:
                gg = G(x, p, t, auxG)
              else:
                gg = G(x, p, p2, t, auxG)
              if AD_efficient:
                dGdx = batchJacobian_AD(gg,x,graphed=True)
              else:
                dGdx = batchJacobian_AD_slow(gg, x, graphed=True)
              if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
                raise RuntimeError(f"Jacobian matrix is NaN")

              x = x.detach()

              torch.set_grad_enabled(False)

        if dGdx.ndim==gg.ndim: # same dimension, must be scalar.
          dx =  (gg/dGdx).detach()
        else:
          dx =  matrixSolve(dGdx, gg).detach()
        x = x - dx
        if useAD_jac:
            torch.set_grad_enabled(True)
        x.requires_grad = True
        if p2 is None:
            gg = G(x, p, t, auxG)
        else:
            gg = G(x, p, p2, t, auxG)
        torch.set_grad_enabled(False)
        resnorm0 = resnorm; ##% old resnorm
        resnorm = torch.linalg.norm(gg, float('inf'),dim= [1]);

    torch.set_grad_enabled(True)
    x = x.detach()
    if not eval:
        if batchP:
          # dGdp is needed only upon convergence.
          if p2 is None:
            if AD_efficient:
                dGdp = batchJacobian_AD(gg, p, graphed=True); dGdp2 = None
            else:
                dGdp = batchJacobian_AD_slow(gg, p, graphed=True);
                dGdp2 = None
          else:
            if AD_efficient:
                dGdp, dGdp2 = batchJacobian_AD(gg, (p,p2),graphed=True)
            else:
                dGdp = batchJacobian_AD_slow(gg, p,graphed=True)# this one is needed only upon convergence.
                dGdp2 = batchJacobian_AD_slow(gg, p2, graphed=True)
            if torch.isnan(dGdp).any() or torch.isinf(dGdp).any() or torch.isnan(dGdp2).any() or torch.isinf(dGdp2).any():
                raise RuntimeError(f"Jacobian matrix is NaN")

        else:
          assert("nonbatchp (like NN) pathway not debugged through yet")

        ctx.save_for_backward(dGdp,dGdp2,dGdx)

    torch.set_grad_enabled(False)
    del gg
    return x

  @staticmethod
  def backward(ctx, dLdx):

    with torch.no_grad():
      dGdp,dGdp2,dGdx = ctx.saved_tensors
      dGdxT = torch.permute(dGdx, (0, 2, 1))
      lambTneg = matrixSolve(dGdxT, dLdx);
      if lambTneg.ndim<=2:
        lambTneg = torch.unsqueeze(lambTneg,2)
      dLdp = -torch.bmm(torch.permute(lambTneg,(0, 2, 1)),dGdp)
      dLdp = torch.squeeze(dLdp,1)
      if dGdp2 is None:
        dLdp2 = None
      else:
        dLdp2 = -torch.bmm(torch.permute(lambTneg,(0, 2, 1)),dGdp2)
        dLdp2 = torch.squeeze(dLdp2,1)
      return dLdp, dLdp2, None,None, None, None, None,None,None
