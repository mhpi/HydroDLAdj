import torch
import numpy as np
import time
import os
import importlib
from HydroDLAdj.nonlinearSolver.MOL import MOL
from HydroDLAdj.nnModel import crit
from HydroDLAdj.utils import rout
device = torch.device("cuda")
dtype=torch.float32


def trainModel(x,
               y,
               z_norm,
               nS,
               nflux,
               nfea,
               nmul,
               model,
               delta_t,
               alpha,
               tdlst,
               startEpoch=1,
               nEpoch=50,
               miniBatch=[100, 365],
               buffTime=0,
               saveFolder=None,
               routn=15,
               dydrop=0.0,
               routDy = False,
               model_name = "HBV_Module"
               ):
    package_name = "HydroDLAdj.HydroModels"
    model_import_string = f"{package_name}.{model_name}"

    try:
        PBMmodel = getattr(importlib.import_module(model_import_string), model_name)
    except ImportError:
        print(f"Failed to import {model_name} from {package_name}")


    lossFun = crit.RmseLossComb(alpha=alpha)
    model = model.cuda()
    lossFun = lossFun.cuda()
    optimizer = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    bs, rho = miniBatch
    ngrid, nt, nx = x.shape
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - bs * (rho) / ngrid / (nt - buffTime))))
    runFile = os.path.join(saveFolder, 'run.csv')
    log_rf = open(runFile, 'a')
    routscaLst = [[0, 2.9], [0, 6.5]]
    print("Start from Epoch ", startEpoch)
    print("Routing days ", routn)
    print("Parameters dropout ", dydrop)
    print("Number of component ", nmul)
    print("Dynamic parameters ", tdlst)
    print("Rounting is Dynamic or not ", routDy)
    #HBV_physics = HBV
    for iEpoch in range(startEpoch, nEpoch + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            tIter = time.time()
            iGrid, iT = randomIndex(ngrid, nt, [bs, rho], bufftime=buffTime)
            xTrain = selectSubset(x, iGrid, iT, rho, bufftime=buffTime)
            yTrain = selectSubset(y, iGrid, iT, rho)
            z_normTrain = selectSubset(z_norm, iGrid, iT, rho, bufftime=buffTime)

            xTrain = xTrain.unsqueeze(1).repeat([1, nmul, 1, 1])

            bsnew = bs * nmul
            fluxSolution_new = torch.zeros((rho, bsnew, nflux)).cuda()
            xTrain = xTrain.view(rho + buffTime, bsnew, -1)

            y0 = torch.zeros((bsnew, nS)).to(device)  # bs*ny

            hbvpara, routpara = model(z_normTrain)  ## LSTM
            if nmul == 1:
                hbvpara = hbvpara.view(rho + buffTime, bsnew, nfea)
                
            else:

                hbvpara = hbvpara.view(rho + buffTime, bs, nfea,nmul)
                hbvpara = hbvpara.permute([0,3,1,2])
                hbvpara = hbvpara.reshape(rho + buffTime, bsnew, nfea)
                if routDy is True:
                    routpara = routpara.view(rho + buffTime, bs,2,nmul)
                    routpara = routpara.permute([0,3,1,2])
                    routpara = routpara.reshape(rho + buffTime, bsnew, 2)

            f_warm_up = PBMmodel(xTrain[:buffTime, :, :], nfea)

            M_warm_up = MOL(f_warm_up, nS, nflux, buffTime, bsDefault=bsnew, mtd=0, dtDefault=delta_t)

            para_warm_up = hbvpara[buffTime - 1, :, :].unsqueeze(0).repeat([buffTime, 1, 1])
            y_warm_up = M_warm_up.nsteps_pDyn(para_warm_up, y0)

            parstaFull = hbvpara[-1, :, :].unsqueeze(0).repeat([rho, 1, 1])  # static matrix
            parhbvFull = torch.clone(parstaFull)
            pmat = torch.ones([1, bsnew]) * dydrop
            for ix in tdlst:
                staPar = parstaFull[:, :, ix - 1]
                dynPar = hbvpara[buffTime:, :, ix - 1]
                drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop some dynamic parameters as static
                comPar = dynPar * (1 - drmask) + staPar * drmask
                parhbvFull[:, :, ix - 1] = comPar
            f = PBMmodel(xTrain[buffTime:, :, :], nfea)
            M = MOL(f, nS, nflux, rho, bsDefault=bsnew, dtDefault=delta_t, mtd=0)
            ### Newton iterations with adjoint
            ySolution = M.nsteps_pDyn(parhbvFull, y_warm_up[-1, :, :])

            tflux = time.time()
            for day in range(0, rho):
                _, flux = f(ySolution[day, :, :], parhbvFull[day, :, :], day)
                fluxSolution_new[day, :, :] = flux * delta_t
            if nmul > 1 and routDy is not True:
                fluxSolution_new = fluxSolution_new.view(rho,nmul,-1,nflux)
                fluxSolution_new = fluxSolution_new.mean(dim=1)

            routa = routscaLst[0][0] + routpara[-1, :, 0] * (routscaLst[0][1] - routscaLst[0][0])
            routb = routscaLst[1][0] + routpara[-1, :, 1] * (routscaLst[1][1] - routscaLst[1][0])

            routa = routa.repeat(rho, 1).unsqueeze(-1)
            routb = routb.repeat(rho, 1).unsqueeze(-1)

            UH = rout.UH_gamma(routa, routb, lenF=routn)  # lenF: folter
            rf = fluxSolution_new.permute([1, 2, 0])  # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = rout.UH_conv(rf, UH).permute([2, 0, 1])
            if nmul > 1 and routDy is True:
                Qsrout = Qsrout.view(rho, nmul, -1, nflux)
                Qsrout = Qsrout.mean(dim=1)

            loss = lossFun(Qsrout[:, :, :], yTrain[:, :, :])
            tback = time.time()
            loss.backward()
            optimizer.step()

            model.zero_grad()
            lossEp = lossEp + loss.item()

            if iIter % 1 == 0:
                IterStr = 'Iter {} of {}: Loss {:.3f} total time {:.2f} fluxes time {:.2f} back time {:.2f}'.format(
                    iIter, nIterEp, loss.item(), time.time() - tIter, time.time() - tflux, time.time() - tback)

                print(IterStr)
                log_rf.write(IterStr + '\n')

                # print loss
        lossEp = lossEp / nIterEp

        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)
        log_rf.write(logStr + '\n')

        modelFile = os.path.join(saveFolder, 'model_Ep' + str(iEpoch) + '.pt')
        torch.save(model, modelFile)
    log_rf.close()




def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if nt <= rho:
        iT.fill(0)

    batchSize = iGrid.shape[0]
    if iT is not None:
        # batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if LCopt is True:
            # used for local calibration kernel: FDC, SMAP...
            if len(x.shape) == 2:
                # Used for local calibration kernel as FDC
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # used for LC-SMAP x=Ngrid*Ntime*Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out

def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT
