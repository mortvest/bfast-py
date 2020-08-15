import numpy as np
import matplotlib.pyplot as plt

from stl import stl
from efp import EFP
from breakpoints import breakpoints


class BFAST():
    def __init__(self, Yt, ti, frequency, h=0.15, season="dummy", max_iter=10,
                 breaks=None, level=0.05):
        # Error catching
        level = np.repeat(level, 2)
        ti = time(Yt)
        f = frequency(Yt) # on cycle every f time points (seasonal cycle)

        Tt = 0
        if season == "harmonic":
            w = 1/f
            tl = np.arange(1, len(Yt) + 1)
            co = np.cos(2 * pi * tl * w)
            si = np.sin(2 * pi * tl * w)
            co2 = np.cos(2 * pi * tl * w * 2)
            si2 = np.sin(2 * pi * tl * w * 2)
            co3 = np.cos(2 * pi * tl * w * 3)
            si3 = np.sin(2 * pi * tl * w * 3)
            smod = Wt ~ co + si + co2 + si2 + co3 + si3

            # Start the iterative procedure and for first iteration St=decompose result
            St = stl(Yt, "periodic")
        elif season == "dummy":
            # Start the iterative procedure and for first iteration St=decompose result
            St = stl(Yt, "periodic")
            D = seasonaldummy(Yt)
            D[rowSums(D) == 0, ] = -1
            smod = Wt ~ -1 + D
        elif season == "none":
            print("No seasonal model will be fitted!")
            St = 0
        else
            raise ValueError("Seasonal model is unknown, use ether 'harmonic' or 'dummy'")

        Vt_bp = 0
        Wt_bp = 0
        CheckTimeTt = 1
        CheckTimeSt = 1
        i = 0

        while (not np.isclose(CheckTimeTt, Vt_bp) or not np.isclose(CheckTimeSt, Wt_bp)) and i < max_iter:
            CheckTimeTt = Vt_bp
            CheckTimeSt = Wt_bp

            ### Change in trend component
            Vt = Yt - St # Deasonalized Time series
            p_Vt = EFP(Vt, ti, h).sctest()
            if p_Vt.p_value <= level[0]:
                bp_Vt = breakpoints(Vt, ti, h=h, breaks=breaks, na_action="exclude")
                nobp_Vt = np.is_nan(breakpoints(bp.Vt)[0])
            else:
                nobp_Vt = True
                bp_Vt = None

            if nobp_Vt:
                ## No Change detected
                fm0 = lm(Vt ~ ti)
                Vt_bp = 0 # no breaks times
                Tt = ts(data=NA,start = ti[1], end = ti[length(ti)],frequency = f) # Data minus trend
                Tt[which(!is.na(Yt))] = fitted(fm0) # Overwrite non-missing with fitted values
                tsp(Tt) = tsp(Yt)
                ci.Vt = None
            else:
                fm1 <- lm(Vt[which(!is.na(Yt))] ~ breakfactor(bp.Vt)/ti[which(!is.na(Yt))] )
                ci.Vt <- confint(bp.Vt, het.err = FALSE)
                Vt.bp <- ci.Vt$confint[, 2]
                # Define empty copy of original time series
                Tt <- ts(data=NA,start = ti[1], end = ti[length(ti)],frequency = f)
                Tt[which(!is.na(Yt))] <- fitted(fm1) # Overwrite non-missing with fitted values
                tsp(Tt) <- tsp(Yt)
            if season == "none":
                Wt = 0
                St = 0
                bp_Wt = None
                ci_Wt = None
                nobp_Wt = True
            else:
                ### Change in seasonal component
                Wt = Yt - Tt
                p_Wt = EFP(smod, h).sctest() # preliminary test
                if p_Wt.p_value <= level[1]:
                    bp_Wt = breakpoints(smod, h=h, breaks=breaks)
                    nobp_Wt = np.isnan(breakpoints(bp_Wt)[0])
                else:
                    nobp_Wt = True
                    bp_Wt = None
                if nobp_Wt:
                    ## No seasonal change detected
                    sm0 = lm(smod)
                    St = ts(data=NA,start = ti[1], end = ti[length(ti)],frequency = f)
                    St[which(!is.na(Yt))] = fitted(sm0) # Overwrite non-missing with fitted values
                    tsp(St) = tsp(Yt)
                    Wt_bp = 0
                    ci_Wt <- None
                else:
                    if season == "dummy":
                        sm1 = lm(Wt ~ -1 + D %in% breakfactor(bp.Wt))
                    if season == "harmonic":
                        sm1 = lm(Wt ~ (co + si + co2 + si2 + co3 + si3) %in% breakfactor(bp.Wt))
                    ci_Wt = confint(bp.Wt, het.err=False)
                    Wt_bp = ci_Wt.confint[, 2]

                    # Define empty copy of original time series
                    St = ts(data=NA,start = ti[1], end = ti[length(ti)],frequency = f)
                    St[which(!is.na(Yt))] = fitted(sm1) # Overwrite non-missing with fitted values
                    tsp(St) = tsp(Yt)
            i += 1

            output[i] = list(Tt = Tt, St = St, Nt = Yt - Tt - St, Vt = Vt, bp.Vt = bp.Vt,
                             Vt.bp = Vt.bp, ci.Vt = ci.Vt, Wt = Wt, bp.Wt = bp.Wt,
                             Wt.bp = Wt.bp, ci.Wt = ci.Wt)

        if not nobp_Vt: # probably only works well for dummy model!
            Vt_nrbp = len(bp_Vt.breakpoints)
            co = coef(fm1) # final fitted trend model
            Mag = matrix(NA, Vt.nrbp, 3)
            for (r in 1:Vt.nrbp):
                if r == 1:
                    y1 = co[1] + co[r + Vt.nrbp + 1] * ti[Vt.bp[r]]
                else:
                    y1 = co[1] + co[r] + co[r + Vt.nrbp + 1] * ti[Vt.bp[r]]

                y2 = (co[1] + co[r + 1]) + co[r + Vt.nrbp + 2] * ti[Vt.bp[r] + 1]

                Mag[r, 0] = y1
                Mag[r, 1] = y2
                Mag[r, 2] = y2 - y1

            index = which.max(abs(Mag[, 3]))
            m_x = rep(Vt.bp[index], 2)
            m_y = c(Mag[index, 1], Mag[index, 2]) #Magnitude position
            Magnitude = Mag[index, 3] # Magnitude of biggest change
            Time = Vt_bp[index]
        else:
            m_x = None
            m_y = None
            Magnitude = 0 # if we do not detect a break then the magnitude is zero
            Time = None # if we do not detect a break then we have no timing of the break
            Mag = 0

        self.Yt = Yt
        self.output = output
        self.nobp = [nobp_Vt, nobp_Wt]
        self.magnitude = Magnitude
        self.mags = mags
        self.time = Time
        self.jump = [ti[m_x], m_y]
