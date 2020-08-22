import logging

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import datasets
import utils
from stl import STL
from efp import EFP
from breakpoints import Breakpoints
from setup import logging_setup


logger = logging.getLogger(__name__)


class BFASTResult():
    def __init__(self, Tt, St, Nt, Vt_bp, Wt_bp):
        self.trend = Tt
        self.season = St
        self.remainder = Nt
        if Vt_bp == np.array([0]).all():
            self.trend_breakpoints = None
        else:
            self.trend_breakpoints = Vt_bp
        if Wt_bp == np.array([0]).all():
            self.season_breakpoints = None
        else:
            self.seasonal_breakpoints = Wt_bp

    def __str__(self):
        st = "Trend:\n{}\n\n".format(self.trend) +\
            "Season:\n{}\n\n".format(self.season) +\
            "Remainder:\n{}\n\n".format(self.remainder) +\
            "Trend Breakpoints:\n{}\n\n".format(self.trend_breakpoints) +\
            "Season Breakpoints:\n{}\n".format(self.season_breakpoints)
        return st


class BFAST():
    def __init__(self, Yt, ti, frequency, h=0.15, season="dummy",
                 max_iter=10, breaks=None, level=0.05, use_mp=True):
        nrow = Yt.shape[0]
        Tt = None
        f = frequency

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

            smod = np.tile(eye_box, (n_boxes, 1))
            smod = smod[:nrow]
        elif season == "none":
            logger.info("'none' season is chosen")
            logger.warning("No sesonal model will be fitted!")
            St = np.zeros(nrow)
        else:
            raise ValueError("Seasonal model is unknown, use 'harmonic', 'dummy' or 'none'")

        Vt_bp = np.array([0])
        Wt_bp = np.array([0])
        CheckTimeTt = np.array([1])
        CheckTimeSt = np.array([1])
        i = 1
        output = None
        nan_map = utils.nan_map(Yt)

        while (Vt_bp != CheckTimeTt).any() or (Wt_bp != CheckTimeSt).any() and i < max_iter:
            logger.info("BFAST iteration #{}".format(i))
            CheckTimeTt = Vt_bp
            CheckTimeSt = Wt_bp

            ### Change in trend component
            with np.errstate(invalid="ignore"):
                Vt = Yt - St  # Deseasonalized Time series
            logger.debug("Vt:\n{}".format(Vt))
            p_Vt = EFP(sm.add_constant(ti), Vt, h).sctest()
            if p_Vt[1] <= level:
                logger.info("Breakpoints in trend detected")
                ti1, Vt1 = utils.omit_nans(ti, Vt)
                logger.info("Finding breakpoints in trend")
                bp_Vt = Breakpoints(sm.add_constant(ti1), Vt1, h=h, breaks=breaks, use_mp=use_mp)
                if bp_Vt.breakpoints is not None:
                    bp_Vt.breakpoints_no_nans = np.array([nan_map[i] for i in bp_Vt.breakpoints])
                    nobp_Vt = False
                else:
                    nobp_Vt = True
            else:
                logger.info("No breakpoints in trend detected")
                nobp_Vt = True
                bp_Vt = None

            logger.info("Fitting linear model for trend")
            if nobp_Vt:
                ## No Change detected
                fm0 = sm.OLS(Vt, ti, missing='drop').fit()
                Vt_bp = np.array([0])  # no breaks times

                Tt = fm0.predict(exog=ti)  # Overwrite non-missing with fitted values
                Tt[np.isnan(Yt)] = np.nan
            else:
                part = bp_Vt.breakfactor()
                X1 = utils.partition(part, ti[~np.isnan(Yt)].flatten())
                y1 = Vt[~np.isnan(Yt)]

                fm1 = sm.OLS(y1, X1, missing='drop').fit()
                # Vt_bp = bp_Vt.breakpoints_no_nans
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
                    logger.info("Breakpoints in season detected")
                    smod1, Wt1 = utils.omit_nans(smod, Wt1)

                    logger.info("Finding breakpoints in season")
                    bp_Wt = Breakpoints(sm.add_constants(mod1), Wt1, h=h, breaks=breaks)
                    if bp_Wt.breakpoints is not None:
                        bp_Wt.breakpoints_no_nans = np.array([nan_map[i] for i in bp_Wt.breakpoints])
                        nobp_Wt = False
                    else:
                        nobp_Wt = True
                else:
                    logger.info("No breakpoints in season detected")
                    nobp_Wt = True
                    bp_Wt = None

                logger.info("Fitting linear model for season")
                if nobp_Wt:
                    ## No seasonal change detected
                    sm0 = sm.OLS(Wt, smod, missing='drop').fit()
                    St = np.repeat(np.nan, nrow)
                    St[~np.isnan(Yt)] = sm0.predict()  # Overwrite non-missing with fitted values
                    Wt_bp = np.array([0])
                else:
                    part = bp_Wt.breakfactor()
                    if season in ["dummy", "harmonic"]:
                        X_sm1 = utils.partition_matrix(part, smod1)

                    sm1 = sm.OLS(Wt1, X_sm1, missing='drop').fit()
                    # Wt_bp = bp_Wt.breakpoints_no_nans
                    Wt_bp = bp_Wt.breakpoints

                    # Define empty copy of original time series
                    St = np.repeat(np.nan, nrow)
                    St[~np.isnan(Yt)] = sm1.predict()  # Overwrite non-missing with fitted values

            with np.errstate(invalid="ignore"):
                Nt = Yt - Tt - St

            i += 1
            output = BFASTResult(Tt, St, Nt, Vt_bp, Wt_bp)

        if not nobp_Vt: # probably only works well for dummy model!
            logger.info("Calculating breakpoint magnitude")
            Vt_nrbp = Vt_bp.shape[0] if Vt_bp is not None else 0
            co = fm1.params  # final fitted trend model
            Mag = np.tile(np.nan, (Vt_nrbp, 3))
            for r in range(Vt_nrbp):
                if r == 0:
                    y1 = co[0] + co[r + Vt_nrbp + 1] * ti[Vt_bp[r]]
                else:
                    y1 = co[0] + co[r] + co[r + Vt_nrbp + 1] * ti[Vt_bp[r]]

                y2 = (co[0] + co[r + 1]) + co[r + Vt_nrbp + 2] * ti[Vt_bp[r] + 1]

                Mag[r, 0] = y1
                Mag[r, 1] = y2
                Mag[r, 2] = y2 - y1
                print("Mag {} = {}".format(r, Mag))

            index = np.argmin(np.abs(Mag[:, 2]) - 1)
            m_x = np.repeat(Vt_bp[index], 2)
            m_y = Mag[index, :2]  # Magnitude position
            Magnitude = Mag[index, 2]  # Magnitude of biggest change
            Time = Vt_bp[index]
        else:
            m_x = None
            m_y = None
            Magnitude = 0  # if we do not detect a break then the magnitude is zero
            Time = None  # if we do not detect a break then we have no timing of the break
            Mag = 0

        self.Yt = Yt
        self.output = output
        self.nobp = (nobp_Vt, nobp_Wt)
        self.magnitude = Magnitude
        self.mags = Mag
        self.time = Time
        self.jump = (ti[m_x], m_y)


if __name__ == "__main__":
    logging_setup()
    y = datasets.ndvi_dates
    x = datasets.ndvi
    freq = datasets.ndvi_freqency

    # v = BFAST(x, y, freq, season="harmonic")
    v = BFAST(x, y, freq, season="dummy")
    # print(v.output)
    print(v.magnitude)
    print(v.time)
    print(v.jump)
