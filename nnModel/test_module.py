import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import importlib
from  HydroDLAdj.nonlinearSolver.MOL import MOL
from HydroDLAdj.utils import rout
from  HydroDLAdj.post import plot, stat
device = torch.device("cuda")
dtype=torch.float32


def testModel( x,
               y,
               z,
               nS,
               nflux,
               nfea,
               nmul,
               model,
               delta_t,
               tdlst,
               bs = 30,
               saveFolder= None,
               routn = 15,
               dydrop = 0.0,
               routDy = False,
               model_name = "HBV_Module",
               useAD_efficient = True,
               ):


    package_name = "HydroDLAdj.HydroModels"
    model_import_string = f"{package_name}.{model_name}"

    try:
        PBMmodel = getattr(importlib.import_module(model_import_string), model_name)
    except ImportError:
        print(f"Failed to import {model_name} from {package_name}")


    ngrid, nt, nx = x.shape
    model = model.cuda()
    model.train(mode=False)
    iS = np.arange(0, ngrid, bs)
    iE = np.append(iS[1:], ngrid)
    routscaLst = [[0, 2.9], [0, 6.5]]
    # forward for each batch
    for i in range(0,  len(iS)):
        bs = iE[i] - iS[i]
        print('batch {}'.format(i))
        xTemp = x[iS[i]:iE[i], :, :]


        xTest = torch.from_numpy( np.swapaxes(xTemp, 1, 0)).float()
        xTest = xTest.cuda()
        zTemp = z[iS[i]:iE[i], :, :]
        zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
        zTest = zTest.cuda()


        hbvpara, routpara = model(zTest)  ## LSTM
        bsnew = bs * nmul
        if nmul == 1:
            hbvpara = hbvpara.view(nt, bsnew, nfea)
        else:

            hbvpara = hbvpara.view(nt, bs, nfea,nmul)
            hbvpara = hbvpara.permute([0,3,1,2])
            hbvpara = hbvpara.reshape(nt, bsnew, nfea)

            if routDy is True:
                routpara = routpara.view(nt, bs, 2,nmul)
                routpara = routpara.permute([0,3,1,2])
                routpara = routpara.reshape(nt, bsnew, 2)



        xTest = xTest.unsqueeze(1).repeat([1,nmul,1,1])

       
        xTest = xTest.view(nt, bsnew, -1)
        y0 = torch.zeros((bsnew, nS)).to(device)  # bs*ny

        fluxSolution_new = torch.zeros((nt, bsnew, nflux)).to(y0)
        fluxSolution_q0 = torch.zeros((nt, bsnew, nflux)).to(y0)
        fluxSolution_q1 = torch.zeros((nt, bsnew, nflux)).to(y0)
        fluxSolution_q2 = torch.zeros((nt, bsnew, nflux)).to(y0)
        fluxSolution_ET = torch.zeros((nt, bsnew, nflux)).to(y0)
        # hbv_para = params[:,:nfea*nmul].detach().requires_grad_(True)



        f = PBMmodel(xTest, nfea)

        M = MOL(f, nS, nflux, nt, bsDefault=bsnew, mtd=0, dtDefault=delta_t,eval = True,AD_efficient=useAD_efficient)

        parstaFull = hbvpara[-1, :, :].unsqueeze(0).repeat([nt, 1, 1])  # static matrix
        parhbvFull = torch.clone(parstaFull)
        pmat = torch.ones([1, bsnew]) * dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix - 1]
            dynPar = hbvpara[:, :, ix-1]
            drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop some dynamic parameters as static
            comPar = dynPar * (1 - drmask) + staPar * drmask
            parhbvFull[:, :, ix - 1] = comPar

        time0 = time.time()
        ySolution = M.nsteps_pDyn(parhbvFull, y0)
        print("nt ", nt)
        print("Time: ", time.time() - time0)
        for day in range(0, nt):
            flux,flux_q0,flux_q1,flux_q2,flux_et = f(ySolution[day, :, :], parhbvFull[day, :, :], day, returnFlux = True)
            fluxSolution_new[day, :, :] = flux * delta_t
            fluxSolution_q0[day, :, :] = flux_q0 * delta_t
            fluxSolution_q1[day, :, :] = flux_q1 * delta_t
            fluxSolution_q2[day, :, :] = flux_q2 * delta_t
            fluxSolution_ET[day, :, :] = flux_et * delta_t

        if nmul > 1 and routDy is not True:
            fluxSolution_new = fluxSolution_new.view(nt, nmul, -1, nflux)
            fluxSolution_new = fluxSolution_new.mean(dim=1)
            fluxSolution_q0 = fluxSolution_q0.view(nt, nmul, -1, nflux)
            fluxSolution_q0 = fluxSolution_q0.mean(dim=1)
            fluxSolution_q1 = fluxSolution_q1.view(nt, nmul, -1, nflux)
            fluxSolution_q1 = fluxSolution_q1.mean(dim=1)
            fluxSolution_q2 = fluxSolution_q2.view(nt, nmul, -1, nflux)
            fluxSolution_q2 = fluxSolution_q2.mean(dim=1)
            fluxSolution_ET = fluxSolution_ET.view(nt, nmul, -1, nflux)
            fluxSolution_ET = fluxSolution_ET.mean(dim=1)

        routa = routscaLst[0][0] + routpara[-1,:, 0] * (routscaLst[0][1] - routscaLst[0][0])
        routb = routscaLst[1][0] + routpara[-1,:, 1] * (routscaLst[1][1] - routscaLst[1][0])
        routa = routa.repeat(nt, 1).unsqueeze(-1)
        routb = routb.repeat(nt, 1).unsqueeze(-1)
        UH = rout.UH_gamma(routa, routb, lenF=routn)  # lenF: folter
        rf = fluxSolution_new.permute([1, 2, 0])  # dim:gage*var*time
        UH = UH.permute([1, 2, 0])  # dim: gage*var*time
        Qsrout = rout.UH_conv(rf, UH).permute([2, 0, 1])
        
        if nmul > 1 and routDy is True:
            Qsrout = Qsrout.view(nt, nmul, -1, nflux)
            Qsrout = Qsrout.mean(dim=1)
        Qsrout = Qsrout.detach().cpu().numpy().swapaxes(0, 1)
        ySolution = ySolution.view(nt, nmul, -1,nS).mean(dim=1)
        SOut = ySolution.detach().cpu().numpy().swapaxes(0, 1)
        Q0 = fluxSolution_q0.detach().cpu().numpy().swapaxes(0, 1)
        Q1 = fluxSolution_q1.detach().cpu().numpy().swapaxes(0, 1)
        Q2 = fluxSolution_q2.detach().cpu().numpy().swapaxes(0, 1)
        ET = fluxSolution_ET.detach().cpu().numpy().swapaxes(0, 1)
        if i == 0:
            yOut = Qsrout
            Spred = SOut
            yQ0 = Q0
            yQ1 = Q1
            yQ2 = Q2
            yET = ET  
        else:
            yOut = np.concatenate((yOut, Qsrout), axis=0)
            Spred = np.concatenate((Spred, SOut), axis=0)            
            yQ0 = np.concatenate((yQ0, Q0), axis=0)
            yQ1 = np.concatenate((yQ1, Q1), axis=0)
            yQ2 = np.concatenate((yQ2, Q2), axis=0)
            yET = np.concatenate((yET, ET), axis=0)
        model.zero_grad()
        torch.cuda.empty_cache()


    evaDict = [stat.statError( yOut[:, -y.shape[1]:, 0],y[:, :, 0])]
    np.save(saveFolder + 'yOut.npy', yOut[:, -y.shape[1]:, 0])
    np.save(saveFolder + 'Spred.npy', Spred[:, -y.shape[1]:, :])
    np.save(saveFolder + 'Q0.npy', yQ0[:, -y.shape[1]:, 0])
    np.save(saveFolder + 'Q1.npy', yQ1[:, -y.shape[1]:, 0])
    np.save(saveFolder + 'Q2.npy', yQ2[:, -y.shape[1]:, 0])
    np.save(saveFolder + 'ET.npy', yET[:, -y.shape[1]:, 0])
    ## Show boxplots of the results
    evaDictLst = evaDict
    keyLst = ['NSE', 'KGE','FLV','FHV','PBiasother', 'lowRMSE', 'highRMSE','midRMSE']
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(evaDictLst)):
            data = evaDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)

    print("NSE,KGE,'PBiaslow','PBiashigh','PBiasother', mean lowRMSE, highRMSE, and midRMSE of all basins in testing period: ", np.nanmedian(dataBox[0][0]),
          np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
          np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]),np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]))



    return




