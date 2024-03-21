import torch
#from HydroDLAdj.Jacobian import batchJacobian_AD
from HydroDLAdj.nonlinearSolver.Jacobian import batchJacobian_AD_loop
#import pydevd
matrixSolve = torch.linalg.solve
class NewtonSolve(torch.autograd.Function):
  # Newton that can custom  gradient tracking for two parameters: p and p2 (both Tensors)
  # if only one is needed, set p2 to None.
  # p2 can be, for example, x^t for Method of Lines time integrator
  # it is a little easier than combining everything inside p
  # if p2 is not None then G should also accept p2: G(x, p, p2, t, auxG)
  # auxG does not track gradient. We must have not auxG variables influenced by p or p2!!!
  @staticmethod
  def forward(ctx, p, p2, t,G, x0=None, auxG=None, batchP=True,eval = False):
    # batchP =True if parameters have a batch dimension
    # with torch.no_grad():

    useAD_jac=True
    if x0 is None and p2 is not None:
      x0 = p2

    x = x0.clone().detach(); i=0;
    max_iter=3; gtol=1e-3;

    if useAD_jac:
        torch.set_grad_enabled(True)

    x.requires_grad = True #p.requires_grad = True

    if p2 is None:
        gg = G(x, p, t, auxG)
    else:
        gg = G(x, p, p2, t, auxG)

  #  dGdx = batchJacobian_AD(gg,x,graphed=True)
    dGdx = batchJacobian_AD_loop(gg, x, graphed=True)
    if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
        raise RuntimeError(f"Jacobian matrix is NaN")
    x = x.detach() # not 100% sure if needed to detach

    torch.set_grad_enabled(False)
    resnorm = torch.linalg.norm(gg, float('inf'),dim= [1]) # calculate norm of the residuals
    resnorm0 = 100*resnorm;


    while ((torch.max(resnorm)>gtol ) and  i<=max_iter):
        i+=1
        if torch.max(resnorm/resnorm0) > 0.2:
              if useAD_jac:
                torch.set_grad_enabled(True)

              x.requires_grad = True #p.requires_grad = True

              if p2 is None:
                gg = G(x, p, t, auxG)
              else:
                gg = G(x, p, p2, t, auxG)

              #dGdx = batchJacobian_AD(gg,x,graphed=True)
              dGdx = batchJacobian_AD_loop(gg, x, graphed=True)
              if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
                raise RuntimeError(f"Jacobian matrix is NaN")

              x = x.detach() # not 100% sure if needed to detach

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
            #dGdp = batchJacobian_AD(gg, p, graphed=True); dGdp2 = None
            dGdp = batchJacobian_AD_loop(gg, p, graphed=True);
            dGdp2 = None
          else:
          #  dGdp, dGdp2 = batchJacobian_AD(gg, (p,p2),graphed=True)
            dGdp = batchJacobian_AD_loop(gg, p,graphed=True)# this one is needed only upon convergence.
            dGdp2 = batchJacobian_AD_loop(gg, p2, graphed=True)
            if torch.isnan(dGdp).any() or torch.isinf(dGdp).any() or torch.isnan(dGdp2).any() or torch.isinf(dGdp2).any():
                raise RuntimeError(f"Jacobian matrix is NaN")
            # it is a tuple of gradient to p and p2 separately, already detached inside
        else:
          assert("nonbatchp (like NN) pathway not debugged through yet")
        # print("day ",t,"Iterations ", i)
        ctx.save_for_backward(dGdp,dGdp2,dGdx)
    # This way, we reduced one forward run. You can also save these two to the CPU if forward run is
    # Alternatively, if memory is a problem, save x and run g during the backward.
    torch.set_grad_enabled(False)
    del gg
    return x

  @staticmethod
  def backward(ctx, dLdx):
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    with torch.no_grad():
      dGdp,dGdp2,dGdx = ctx.saved_tensors
      dGdxT = torch.permute(dGdx, (0, 2, 1))
      lambTneg = matrixSolve(dGdxT, dLdx);
      if lambTneg.ndim<=2:
        lambTneg = torch.unsqueeze(lambTneg,2)
      dLdp = -torch.bmm(torch.permute(lambTneg,(0, 2, 1)),dGdp)
      dLdp = torch.squeeze(dLdp,1) # ADHOC!! DON"T KNOW WHY!!
      if dGdp2 is None:
        dLdp2 = None
      else:
        dLdp2 = -torch.bmm(torch.permute(lambTneg,(0, 2, 1)),dGdp2)
        dLdp2 = torch.squeeze(dLdp2,1) # ADHOC!! DON"T KNOW WHY!!
      return dLdp, dLdp2, None,None, None, None, None,None
