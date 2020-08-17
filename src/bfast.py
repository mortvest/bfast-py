import logging

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

from stl import STL
from efp import EFP
from breakpoints import Breakpoints
from setup import logging_setup
import datasets
import utils


logger = logging.getLogger(__name__)


class BFASTResult():
    def __init__(self, Tt, St, Nt, Vt_bp, Wt_bp):
        self.trend = Tt
        self.season = St
        self.remainder = Nt
        self.trend_breakpoints = Vt_bp
        self.season_breakpoints = Wt_bp


class BFAST():
    def __init__(self, Yt, ti, frequency, h=0.15, season="dummy",
                 max_iter=10, breaks=None, level=0.05, use_mp=True):
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
            # smod = np.column_stack((np.ones(nrow), co, si, co2, si2, co3, si3))
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
        output = None
        nan_map = utils.nan_map(Yt)

        while (not np.isclose(CheckTimeTt, Vt_bp).all()
               or not np.isclose(CheckTimeSt, Wt_bp).all()) and i < max_iter:
            CheckTimeTt = Vt_bp
            CheckTimeSt = Wt_bp

            ### Change in trend component
            with np.errstate(invalid="ignore"):
                Vt = Yt - St  # Deseasonalized Time series
            logger.info("Vt:\n{}".format(Vt))
            p_Vt = EFP(sm.add_constant(ti), Vt, h).sctest()
            if p_Vt[1] <= level:
                ti1, Vt1 = utils.omit_nans(ti, Vt)
                bp_Vt = Breakpoints(sm.add_constant(ti1), Vt1, h=h, breaks=breaks, use_mp=use_mp)
                if bp_VT.breakpoints is not None:
                    bp_Vt.breakpoints = np.array([nan_map[i] for i in bp_Vt.breakpoints])
                    nobp_Vt = False
                else:
                    nobp_Vt = True
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
                part = bp_Vt.breakfactor()
                X1 = utils.partition(part, ti[~np.isnan(Yt)].flatten())
                y1 = Vt[~np.isnan(Yt)]

                fm1 = sm.OLS(y1, X1, missing='drop').fit()
                Vt_bp = bp_Vt.breakpoints

                Tt = np.repeat(np.nan, ti.shape[0])
                Tt[~np.isnan(Yt)] = fm1.predict()

            if season == "none":
                Wt = np.zeros(nrow).astype(float)
                St = np.zeros(nrow).astype(float)
                bp_Wt = None
                nobp_Wt = True
            else:
                ### Change in seasonal component
                with np.errstate(invalid="ignore"):
                    Wt = Yt - Tt
                p_Wt = EFP(sm.add_constant(smod), Wt, h).sctest()  # preliminary test
                if p_Wt[1] <= level:
                    smod1, Wt1 = utils.omit_nans(smod, Wt1)
                    bp_Wt = Breakpoints(sm.add_constants(mod1), Wt1, h=h, breaks=breaks)
                    if bp_Wt.breakpoints is not None:
                        bp_Wt.breakpoints = np.array([nan_map[i] for i in bp_Wt.breakpoints])
                        nobp_Wt = False
                    else:
                        nobp_Wt = True
                else:
                    nobp_Wt = True
                    bp_Wt = None

                if nobp_Wt:
                    ## No seasonal change detected
                    sm0 = sm.OLS(Wt, smod, missing='drop').fit()
                    St = np.repeat(np.nan, nrow)
                    St[~np.isnan(Yt)] = sm0.predict()  # Overwrite non-missing with fitted values
                    Wt_bp = 0
                else:
                    part = bp_Wt.breakfactor()
                    if season == "dummy":
                        X_sm1 = utils.partition_matrix(part, smod1)
                    if season == "harmonic":
                        X_sm1 = utils.partition_matrix(part, smod1)

                    sm1 = sm.OLS(Wt1, X_sm1, missing='drop').fit()
                    Wt_bp = bp_Wt.breakpoints

                    # Define empty copy of original time series
                    St = np.repeat(np.nan, nrow)
                    St[~np.isnan(Yt)] = sm1.predict()  # Overwrite non-missing with fitted values

            with np.errstate(invalid="ignore"):
                Nt = Yt - Tt - St

            output = BFASTResult(Tt, St, Nt, Vt_bp, Wt_bp)

        # if not nobp_Vt: # probably only works well for dummy model!
        #     Vt_nrbp = Vt_bp.shape[0] if Vt_bp is not None else 0
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

        self.Yt = Yt
        self.output = output
        self.nobp = (nobp_Vt, nobp_Wt)
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
