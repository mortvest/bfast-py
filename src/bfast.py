import numpy as np
import statsmodels.api as sm

import utils
from datasets import *
from stl import STL
from efp import EFP
from breakpoints import Breakpoints


class BFASTResult():
    def __init__(self, Tt, St, Nt, Vt_bp, Wt_bp):
        self.trend = Tt
        self.season = St
        self.remainder = Nt
        if (Vt_bp == np.array([0])).all():
            self.trend_breakpoints = None
        else:
            self.trend_breakpoints = Vt_bp
        if Wt_bp == np.array([0]).all():
            self.season_breakpoints = None
        else:
            self.season_breakpoints = Wt_bp

    def __str__(self):
        st = "Trend:\n{}\n\n".format(self.trend) +\
            "Season:\n{}\n\n".format(self.season) +\
            "Remainder:\n{}\n\n".format(self.remainder) +\
            "Trend Breakpoints:\n{}\n\n".format(self.trend_breakpoints) +\
            "Season Breakpoints:\n{}\n".format(self.season_breakpoints)
        return st


class BFAST(utils.LoggingBase):
    """
    Iterative break detection in seasonal and trend component of a time
    series. Seasonal breaks is a function that combines the iterative
    decomposition of time series into trend, seasonal and remainder
    components with significant break detection in the decomposed
    components of the time series.
    """
    def __init__(self,
                 Yt,
                 ti,
                 frequency,
                 h=0.15,
                 season="dummy",
                 max_iter=10,
                 max_breaks=None,
                 level=0.05,
                 use_mp=True,
                 verbosity=0):
        super().__init__(verbosity)
        nrow = Yt.shape[0]
        Tt = None
        f = frequency

        if season == "harmonic":
            self.logger.info("'harmonic' season is chosen")
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
            self.logger.info("Applying STL")
            St = STL(Yt, f, periodic=True).seasonal
            self.logger.debug("St set to\n{}".format(St))
        elif season == "dummy":
            self.logger.info("'dummy' season is chosen")
            # Start the iterative procedure and for first iteration St=decompose result
            self.logger.info("Applying STL")
            St = STL(Yt, f, periodic=True).seasonal

            eye_box = np.row_stack((np.eye(f - 1), np.repeat(-1, f - 1)))
            n_boxes = int(np.ceil(nrow / f))

            smod = np.tile(eye_box, (n_boxes, 1))
            smod = smod[:nrow]
            smod = sm.add_constant(smod)
        elif season == "none":
            self.logger.info("'none' season is chosen")
            self.logger.warning("No sesonal model will be fitted!")
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
            self.logger.info("BFAST iteration #{}".format(i))
            CheckTimeTt = Vt_bp
            CheckTimeSt = Wt_bp

            ### Change in trend component
            with np.errstate(invalid="ignore"):
                Vt = Yt - St  # Deseasonalized Time series
            self.logger.debug("Vt:\n{}".format(Vt))
            p_Vt = EFP(sm.add_constant(ti), Vt, h, verbosity=self.verbosity).sctest()
            if p_Vt[1] <= level:
                self.logger.info("Breakpoints in trend detected")
                ti1, Vt1 = utils.omit_nans(ti, Vt)
                self.logger.info("Finding breakpoints in trend")
                bp_Vt = Breakpoints(sm.add_constant(ti1), Vt1, h=h, max_breaks=max_breaks, use_mp=use_mp)
                if bp_Vt.breakpoints is not None:
                    bp_Vt.breakpoints_no_nans = np.array([nan_map[i] for i in bp_Vt.breakpoints])
                    nobp_Vt = False
                else:
                    nobp_Vt = True
            else:
                self.logger.info("No breakpoints in trend detected")
                nobp_Vt = True
                bp_Vt = None

            self.logger.info("Fitting linear model for trend")
            if nobp_Vt:
                ## No Change detected
                fm0 = sm.OLS(Vt, ti, missing='drop').fit()
                Vt_bp = np.array([0])  # no breaks times

                Tt = fm0.predict(exog=ti)  # Overwrite non-missing with fitted values
                Tt[np.isnan(Yt)] = np.nan
            else:
                part = bp_Vt.breakfactor()
                X1 = utils.partition_matrix(part, sm.add_constant(ti[~np.isnan(Yt)]))
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
                p_Wt = EFP(smod, Wt, h, verbosity=self.verbosity).sctest()  # preliminary test
                if p_Wt[1] <= level:
                    self.logger.info("Breakpoints in season detected")
                    smod1, Wt1 = utils.omit_nans(smod, Wt)

                    self.logger.info("Finding breakpoints in season")
                    bp_Wt = Breakpoints(smod1, Wt1, h=h, max_breaks=max_breaks)
                    if bp_Wt.breakpoints is not None:
                        bp_Wt.breakpoints_no_nans = np.array([nan_map[i] for i in bp_Wt.breakpoints])
                        nobp_Wt = False
                    else:
                        nobp_Wt = True
                else:
                    self.logger.info("No breakpoints in season detected")
                    nobp_Wt = True
                    bp_Wt = None

                self.logger.info("Fitting linear model for season")
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
            self.logger.info("Calculating breakpoint magnitude")
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

        self.n_iter = i
        self.Yt = Yt
        self.output = output
        self.nobp = (nobp_Vt, nobp_Wt)
        self.magnitude = Magnitude
        self.mags = Mag
        self.time = Time
        self.jump = ti[m_x], m_y


def run_test(y, x, f, season, level=0.05, h=0.15, max_iter=10, verbosity=0):
    v = BFAST(y, x, f, season=season, level=level, h=h, max_iter=max_iter, verbosity=verbosity)
    Vt_bp = v.output.trend_breakpoints
    Vt_dates = x[Vt_bp] if (Vt_bp is not None) else None

    Wt_bp = v.output.season_breakpoints
    Wt_dates = x[Wt_bp] if (Wt_bp is not None) else None

    # print("n_iters", v.n_iter)
    print("Trend breakpoints", Vt_bp)
    # print("Trend time", Vt_dates)
    # print("Season breakpoints", Wt_bp)
    # print("Season time", Wt_dates)


if __name__ == "__main__":
    # logging_setup()
    # run_test(nile, nile_dates, None, "none")
    # run_test(simts_sum, simts_dates, simts_freq, "harmonic", level=0.35, h=0.3, max_iter=2)
    # run_test(ndvi, ndvi_dates, ndvi_freq, "dummy")
    run_test(harvest, harvest_dates, harvest_freq, "harmonic", verbosity=0)
