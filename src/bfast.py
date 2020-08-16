import logging

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

from stl import STL
from efp import EFP
from breakpoints import Breakpoints
import datasets
from setup import logging_setup


logger = logging.getLogger(__name__)


class BFAST():
    def __init__(self, Yt, ti, frequency, h=0.15, season="dummy",
                 max_iter=10, breaks=None, level=0.05):
        nrow = Yt.shape[0]
        Tt = 0
        f = frequency
        output = []
        if season == "harmonic":
            logger.info("'harmonic' season is chosen")
            w = 1/f
            tl = np.arange(1, Yt.shape[0] + 1)
            co = np.cos(2 * np.pi * tl * w)
            si = np.sin(2 * np.pi * tl * w)
            co2 = np.cos(2 * np.pi * tl * w * 2)
            si2 = np.sin(2 * np.pi * tl * w * 2)
            co3 = np.cos(2 * np.pi * tl * w * 3)
            si3 = np.sin(2 * np.pi * tl * w * 3)

            # smod = Wt ~ co + si + co2 + si2 + co3 + si3
            smod = np.column_stack((co, si, co2, si2, co3, si3))

            # Start the iterative procedure and for first iteration St=decompose result
            St = STL(Yt, f, periodic=True).seasonal
            logger.debug("St set to\n{}".format(St))
        elif season == "dummy":
            logger.info("'dummy' season is chosen")
            # Start the iterative procedure and for first iteration St=decompose result
            St = STL(Yt, f, periodic=True).seasonal

            eye_box = np.row_stack((np.eye(f - 1), np.repeat(-1, f - 1)))
            n_boxes = int(np.ceil(nrow / f))
            D = np.tile(eye_box, (n_boxes, 1))
            D = D[:nrow]

            # smod = Wt ~ -1 + D
            smod = np.column_stack((np.repeat(-1.0, nrow), D))
        elif season == "none":
            logger.info("'none' season is chosen")
            logger.warning("No sesonal model will be fitted!")
            St = np.zeros(nrow)
        else:
            raise ValueError("Seasonal model is unknown, use 'harmonic', 'dummy' or 'none'")

        Vt_bp = 0
        Wt_bp = 0
        CheckTimeTt = 1
        CheckTimeSt = 1
        i = 0

        while (not np.isclose(CheckTimeTt, Vt_bp).all()
               or not np.isclose(CheckTimeSt, Wt_bp).all()) and i < max_iter:
            CheckTimeTt = Vt_bp
            CheckTimeSt = Wt_bp

            ### Change in trend component
            with np.errstate(invalid="ignore"):
                Vt = Yt - St  # Deseasonalized Time series
            logger.info("Vt:\n{}".format(Vt))
            p_Vt = EFP(ti, Vt, h).sctest()
            if p_Vt[1] <= level:
                bp_Vt = Breakpoints(Vt, ti, h=h, breaks=breaks)
                nobp_Vt = bp_Vt.breakpoints is None
            else:
                nobp_Vt = True
                bp_Vt = None

            if nobp_Vt:
                ## No Change detected
                fm0 = sm.OLS(Vt, ti, missing='drop').fit()
                Vt_bp = 0  # no breaks times
                Tt = fm0.predict(exog=ti)  # Overwrite non-missing with fitted values
                Tt[np.isnan(Yt)] = np.nan
            else:
                X1 = bp_Vt.breakfactor / ti[~np.isnan(Yt)]
                Y1 = Vt[~np.isnan(Yt)]
                fm1 = sm.OLS(Y1, X1, missing='drop')
                Vt_bp = bp_Vt.breakpoints
                # Define empty copy of original time series
                Tt = fm1.predict(exog=ti)  # Overwrite non-missing with fitted values
                Tt[np.isnan(Yt)] = np.nan
            exit()
        #     if season == "none":
        #         Wt = 0
        #         St = 0
        #         bp_Wt = None
        #         ci_Wt = None
        #         nobp_Wt = True
        #     else:
        #         ### Change in seasonal component
        #         Wt = Yt - Tt
        #         p_Wt = EFP(smod, h).sctest()  # preliminary test
        #         if p_Wt.p_value <= level:
        #             bp_Wt = breakpoints(smod, h=h, breaks=breaks)
        #             nobp_Wt = np.isnan(breakpoints(bp_Wt)[0])
        #         else:
        #             nobp_Wt = True
        #             bp_Wt = None
        #         if nobp_Wt:
        #             ## No seasonal change detected
        #             sm0 = lm(smod)
        #             St = ts(data=NA,start = ti[1], end = ti[length(ti)],frequency = f)
        #             St[which(!is.na(Yt))] = fitted(sm0) # Overwrite non-missing with fitted values
        #             tsp(St) = tsp(Yt)
        #             Wt_bp = 0
        #             ci_Wt <- None
        #         else:
        #             if season == "dummy":
        #                 sm1 = lm(Wt ~ -1 + D %in% breakfactor(bp.Wt))
        #             if season == "harmonic":
        #                 sm1 = lm(Wt ~ (co + si + co2 + si2 + co3 + si3) %in% breakfactor(bp.Wt))
        #             ci_Wt = confint(bp.Wt, het.err=False)
        #             Wt_bp = ci_Wt.confint[, 2]

        #             # Define empty copy of original time series
        #             St = ts(data=NA,start = ti[1], end = ti[length(ti)],frequency = f)
        #             St[which(!is.na(Yt))] = fitted(sm1) # Overwrite non-missing with fitted values
        #             tsp(St) = tsp(Yt)
        #     i += 1

        #     output[i] = list(Tt = Tt, St = St, Nt = Yt - Tt - St, Vt = Vt, bp.Vt = bp.Vt,
        #                      Vt.bp = Vt.bp, ci.Vt = ci.Vt, Wt = Wt, bp.Wt = bp.Wt,
        #                      Wt.bp = Wt.bp, ci.Wt = ci.Wt)

        # if not nobp_Vt: # probably only works well for dummy model!
        #     Vt_nrbp = len(bp_Vt.breakpoints)
        #     co = coef(fm1) # final fitted trend model
        #     Mag = matrix(NA, Vt.nrbp, 3)
        #     for (r in 1:Vt.nrbp):
        #         if r == 1:
        #             y1 = co[1] + co[r + Vt.nrbp + 1] * ti[Vt.bp[r]]
        #         else:
        #             y1 = co[1] + co[r] + co[r + Vt.nrbp + 1] * ti[Vt.bp[r]]

        #         y2 = (co[1] + co[r + 1]) + co[r + Vt.nrbp + 2] * ti[Vt.bp[r] + 1]

        #         Mag[r, 0] = y1
        #         Mag[r, 1] = y2
        #         Mag[r, 2] = y2 - y1

        #     index = which.max(abs(Mag[, 3]))
        #     m_x = rep(Vt.bp[index], 2)
        #     m_y = c(Mag[index, 1], Mag[index, 2]) #Magnitude position
        #     Magnitude = Mag[index, 3] # Magnitude of biggest change
        #     Time = Vt_bp[index]
        # else:
        #     m_x = None
        #     m_y = None
        #     Magnitude = 0 # if we do not detect a break then the magnitude is zero
        #     Time = None # if we do not detect a break then we have no timing of the break
        #     Mag = 0

        # self.Yt = Yt
        # self.output = output
        # self.nobp = [nobp_Vt, nobp_Wt]
        # self.magnitude = Magnitude
        # self.mags = mags
        # self.time = Time
        # self.jump = [ti[m_x], m_y]

if __name__ == "__main__":
    logging_setup()
    y = datasets.ndvi_dates
    x = datasets.ndvi
    freq = datasets.ndvi_freqency

    v = BFAST(x, y, freq, season="harmonic")
