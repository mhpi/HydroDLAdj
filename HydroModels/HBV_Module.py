import torch
import torch.nn as nn

class HBV_Module(nn.Module):
    def __init__(self, climate_data,nfea):
        super().__init__()
        self.climate_data = climate_data
        self.nfea = nfea

    def forward(self, y,theta,t,returnFlux=False,aux=None):
        ##parameters
        bs = theta.shape[0]

        nfea = self.nfea
        theta = theta.view(bs,nfea)
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5]] # HBV para

        Beta = parascaLst[0][0] + theta[:,0]*(parascaLst[0][1]-parascaLst[0][0])
        FC = parascaLst[1][0] + theta[:,1]*(parascaLst[1][1]-parascaLst[1][0])
        K0 = parascaLst[2][0] + theta[:,2]*(parascaLst[2][1]-parascaLst[2][0])
        K1 = parascaLst[3][0] + theta[:,3]*(parascaLst[3][1]-parascaLst[3][0])
        K2 = parascaLst[4][0] + theta[:,4]*(parascaLst[4][1]-parascaLst[4][0])
        LP = parascaLst[5][0] + theta[:,5]*(parascaLst[5][1]-parascaLst[5][0])
        PERC = parascaLst[6][0] + theta[:,6]*(parascaLst[6][1]-parascaLst[6][0])
        UZL = parascaLst[7][0] + theta[:,7]*(parascaLst[7][1]-parascaLst[7][0])
        TT = parascaLst[8][0] + theta[:,8]*(parascaLst[8][1]-parascaLst[8][0])
        CFMAX = parascaLst[9][0] + theta[:,9]*(parascaLst[9][1]-parascaLst[9][0])
        CFR = parascaLst[10][0] + theta[:,10]*(parascaLst[10][1]-parascaLst[10][0])
        CWH = parascaLst[11][0] + theta[:,11]*(parascaLst[11][1]-parascaLst[11][0])
        BETAET = parascaLst[12][0] + theta[:,12]*(parascaLst[12][1]-parascaLst[12][0])



        PRECS = 0
        ##% stores
        SNOWPACK = torch.clamp(y[:,0] , min=PRECS)  #SNOWPACK
        MELTWATER = torch.clamp(y[:,1], min=PRECS)  #MELTWATER
        SM = torch.clamp(y[:,2], min=1e-8)   #SM
        SUZ = torch.clamp(y[:,3] , min=PRECS) #SUZ
        SLZ = torch.clamp(y[:,4], min=PRECS)   #SLZ
        dS = torch.zeros(y.shape[0],y.shape[1]).to(y)
        fluxes = torch.zeros((y.shape[0],1)).to(y)

        climate_in = self.climate_data[int(t),:,:];   ##% climate at this step
        P  = climate_in[:,0];
        Ep = climate_in[:,2];
        T  = climate_in[:,1];

        ##% fluxes functions
        flux_sf   = self.snowfall(P,T,TT);
        flux_refr = self.refreeze(CFR,CFMAX,T,TT,MELTWATER);
        flux_melt = self.melt(CFMAX,T,TT,SNOWPACK);
        flux_rf   = self.rainfall(P,T,TT);
        flux_Isnow   =  self.Isnow(MELTWATER,CWH,SNOWPACK);
        flux_PEFF   = self.Peff(SM,FC,Beta,flux_rf,flux_Isnow);
        flux_ex   = self.excess(SM,FC);
        flux_et   = self.evap(SM,FC,LP,Ep,BETAET);
        flux_perc = self.percolation(PERC,SUZ);
        flux_q0   = self.interflow(K0,SUZ,UZL);
        flux_q1   = self.baseflow(K1,SUZ);
        flux_q2   = self.baseflow(K2,SLZ);


        #% stores ODEs
        dS[:,0] = flux_sf + flux_refr - flux_melt
        dS[:,1] = flux_melt - flux_refr - flux_Isnow
        dS[:,2] = flux_Isnow + flux_rf - flux_PEFF - flux_ex - flux_et 
        dS[:,3] = flux_PEFF + flux_ex - flux_perc - flux_q0 - flux_q1
        dS[:,4] = flux_perc - flux_q2 

        fluxes[:,0] =flux_q0 + flux_q1 + flux_q2

        if returnFlux:
            return fluxes,flux_q0.unsqueeze(-1),flux_q1.unsqueeze(-1),flux_q2.unsqueeze(-1),flux_et.unsqueeze(-1)
        else:
            return dS,fluxes





    def rechange(self, C,NDC,FC,SM,SLZ):
        return  C*SLZ*(1.0-torch.clamp(SM/(NDC*FC),max = 1.0))

 

    def snowfall(self,P,T,TT):
        return torch.mul(P, (T < TT))

    def refreeze(self,CFR,CFMAX,T,TT,MELTWATER):
        refreezing = CFR * CFMAX * (TT - T)
        refreezing = torch.clamp(refreezing, min=0.0)
        return torch.min(refreezing, MELTWATER)

    def melt(self,CFMAX,T,TT,SNOWPACK):
        melt = CFMAX * (T - TT)
        melt = torch.clamp(melt, min=0.0)
        return torch.min(melt, SNOWPACK)

    def rainfall(self,P,T,TT):

        return torch.mul(P, (T >= TT))
    def Isnow(self,MELTWATER,CWH,SNOWPACK):
        tosoil = MELTWATER - (CWH * SNOWPACK)
        tosoil = torch.clamp(tosoil, min=0.0)
        return tosoil

    def Peff(self,SM,FC,Beta,flux_rf,flux_Isnow):
        soil_wetness = (SM / FC) ** Beta
        soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
        return (flux_rf + flux_Isnow) * soil_wetness

    def excess(self,SM,FC):
        excess = SM - FC
        return  torch.clamp(excess, min=0.0)


    def evap(self,SM,FC,LP,Ep,BETAET):
        evapfactor = (SM / (LP * FC)) ** BETAET
        evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
        ETact = Ep * evapfactor
        return torch.min(SM, ETact)

    def interflow(self,K0,SUZ,UZL):
        return K0 * torch.clamp(SUZ - UZL, min=0.0)
    def percolation(self,PERC,SUZ):
        return torch.min(SUZ, PERC)

    def baseflow(self,K,S):
        return K * S
