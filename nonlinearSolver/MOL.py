import torch

from HydroDLAdj.nonlinearSolver.NewtonSolve import NewtonSolve

newtonAdj = NewtonSolve.apply
class MOL(torch.nn.Module):
  # Method of Lines time integrator as a nonlinear equation G(x, p, xt, t, auxG)=0.
  # rhs is preloaded at construct and is the equation for the right hand side of the equation.
    def __init__(self, rhsFunc,ny,nflux,rho, bsDefault =1 , mtd = 0, dtDefault=0, solveAdj = newtonAdj,eval = False,AD_efficient=True):
        super(MOL, self).__init__()
        self.mtd = mtd # time discretization method. =0 for backward Euler
        self.rhs = rhsFunc
        self.delta_t = dtDefault
        self.bs = bsDefault
        self.ny = ny
        self.nflux = nflux
        self.rho = rho
        self.solveAdj = solveAdj
        self.eval = eval
        self.AD_efficient = AD_efficient

    def forward(self, x, p, xt, t, auxG): # take one step
        # xt is x^{t}. trying to solve for x^{t+1}
        dt, aux = auxG # expand auxiliary data

        if self.mtd == 0: # backward Euler
          rhs,_ = self.rhs(x, p, t, aux) # should return [nb,ng]
          gg = (x - xt)/dt - rhs
        elif self.mtd == 1: # Crank Nicholson
          rhs,_  = self.rhs(x, p, t, aux) # should return [nb,ng]
          rhst,_ = self.rhs(xt, p, t, aux) # should return [nb,ng]
          gg = (x - xt)/dt - (rhs+rhst)*0.5
        return gg

    def nsteps_pDyn(self,pDyn,x0):
        bs = self.bs
        ny = self.ny
        delta_t = self.delta_t
        rho = self.rho
        ySolution = torch.zeros((rho,bs,ny)).to(pDyn)
        ySolution[0,:,:] = x0

        xt=x0.clone().requires_grad_()

        auxG = (delta_t, None)

        for t in range(rho):
            p = pDyn[t,:,:]

            x = self.solveAdj(p, xt,t, self.forward, None, auxG,True, self.eval,self.AD_efficient)

            ySolution[t,:,:]  = x
            xt = x

        return ySolution
