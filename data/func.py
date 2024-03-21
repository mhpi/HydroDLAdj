import pickle
import numpy as np

def basinNorm_mmday(x, basinarea,  toNorm):
  nd = len(x.shape)
  if nd == 3 and x.shape[2] == 1:
    x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
  temparea = np.tile(basinarea, (1, x.shape[1]))
  #tempprep = np.tile(meanprep, (1, x.shape[1]))
  if toNorm is True:
    flow = (x * 0.0283168 * 3600 * 24) / (
            (temparea * (10 ** 6))* (10 ** (-3))
    )  # (m^3/day)/(m^3/day)
  else:

    flow = (
            x
            * ((temparea * (10 ** 6))* (10 ** (-3)))
            / (0.0283168 * 3600 * 24)
    )
  if nd == 3:
    flow = np.expand_dims(flow, axis=2)
  return flow

# This part defines a hydrology-specific Scaler class that works similar as sklearn.MinMaxScaler
# getStatDic, calcStat, calcStatgamma are supporting functions
def getStatDic(log_norm_cols, attrLst=None, attrdata=None, seriesLst=None, seriesdata=None):
  statDict = dict()
  # series data
  if seriesLst is not None:
    for k in range(len(seriesLst)):
      var = seriesLst[k]
      if var in log_norm_cols:
        statDict[var] = calcStatgamma(seriesdata[:, :, k])
      else:
        statDict[var] = calcStat(seriesdata[:, :, k])

  # const attribute
  if attrLst is not None:
    for k in range(len(attrLst)):
      var = attrLst[k]
      statDict[var] = calcStat(attrdata[:, k])
  return statDict

def calcStat(x):
  a = x.flatten()
  b = a[~np.isnan(a)]
  p10 = np.percentile(b, 10).astype(float)
  p90 = np.percentile(b, 90).astype(float)
  mean = np.mean(b).astype(float)
  std = np.std(b).astype(float)
  if std < 0.001:
    std = 1
  return [p10, p90, mean, std]

def calcStatgamma(x):  # for daily streamflow and precipitation
  a = x.flatten()
  b = a[~np.isnan(a)]  # kick out Nan
  b = np.log10(
    np.sqrt(b) + 0.1
  )  # do some tranformation to change gamma characteristics
  p10 = np.percentile(b, 10).astype(float)
  p90 = np.percentile(b, 90).astype(float)
  mean = np.mean(b).astype(float)
  std = np.std(b).astype(float)
  if std < 0.001:
    std = 1
  return [p10, p90, mean, std]

def transNormbyDic( x_in, var_lst, stat_dict, log_norm_cols, to_norm):
  if type(var_lst) is str:
    var_lst = [var_lst]
  x = x_in.copy()
  out = np.full(x.shape, np.nan)
  for k in range(len(var_lst)):
    var = var_lst[k]
    stat = stat_dict[var]
    if to_norm is True:
      if len(x.shape) == 3:
        if var in log_norm_cols:
          x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
        out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
      elif len(x.shape) == 2:
        if var in log_norm_cols:
          x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
        out[:, k] = (x[:, k] - stat[2]) / stat[3]
    else:
      if len(x.shape) == 3:
        out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
        if var in log_norm_cols:
          out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
      elif len(x.shape) == 2:
        out[:, k] = x[:, k] * stat[3] + stat[2]
        if var in log_norm_cols:
          out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
  return out



class HydroScaler:
  def __init__(self, attrLst, seriesLst, xNanFill,log_norm_cols):
    self.log_norm_cols = log_norm_cols
    self.attrLst = attrLst
    self.seriesLst = seriesLst
    self.stat_dict = None
    self.xNanFill = xNanFill

  def fit(self, attrdata, seriesdata):
    self.stat_dict = getStatDic(
      log_norm_cols=self.log_norm_cols,
      attrLst=self.attrLst,
      attrdata=attrdata,
      seriesLst=self.seriesLst,
      seriesdata=seriesdata,
    )

  def transform(self, data, var_list,):

    norm_data = transNormbyDic(
      data, var_list, self.stat_dict, log_norm_cols = self.log_norm_cols, to_norm=True)

    return norm_data

  def fit_transform(self, attrdata, seriesdata):
    self.fit(attrdata, seriesdata)
    attr_norm = self.transform(attrdata, self.attrLst)
    series_norm = self.transform(seriesdata, self.seriesLst)
    return attr_norm, series_norm



