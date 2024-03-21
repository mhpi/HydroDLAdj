import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask



generator = torch.Generator(device="cuda")
generator.manual_seed(42)
class lstmModel(torch.nn.Module):
    # class for LSTM to generate multiple components and static parameters
    def __init__(self, ninv, nfea, nmul, hiddeninv,  drinv=0.5,routDy = False):

        super(lstmModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.nmul = nmul
        # get the total number of parameters
        nhbvpm = nfea*nmul
        if routDy is True:
            nroutpm = 2*nmul
        else:
            nroutpm = 2
        ntp = nhbvpm + nroutpm

        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        self.nhbvpm = nhbvpm
        self.nroutpm = nroutpm
    def forward(self, z, doDropMC=False):
        Gen = self.lstminv(z)
        Params0 = Gen[:, :, :] # the last time step as learned parameters

        # print(Params0)
        hbvpara0 = Params0[:, :, 0:self.nhbvpm]  ##nt bs fea*nmul
        hbvpara = torch.sigmoid(hbvpara0)
        routpara0 = Params0[:, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # dim: [Ngage, nmul*2] or [Ngage, 2]
        routpara = torch.sigmoid(routpara0)

        return hbvpara, routpara

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)
        return out


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.randn(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.randn(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.randn(hiddenSize * 4))
        self.b_hh = Parameter(torch.randn(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv,generator=generator)


    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        output, hy, cy, reserve, new_weight_buf = torch._C._VariableFunctions._cudnn_rnn(
            input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
            self.hiddenSize, 0, 1, False, 0, self.training, False, (), None)

        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]

