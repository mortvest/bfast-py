import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime, timedelta


def r_style_interval(from_tuple, end_tuple, frequency):
    """
    create time interval using R-style double-tuple notation
    """
    from_year, from_seg = from_tuple
    end_year, end_seg = end_tuple
    n = (end_year - from_year + 1) * frequency
    full_range = np.linspace(from_year, end_year + 1, num=n, endpoint=False)
    real_range = full_range[(from_seg - 1):n - (frequency - end_seg)]
    return real_range


data_folder = "../data/"


"""
R dataset: Average Yearly Temperatures in New Haven
"""
nhtemp = np.array([49.9, 52.3, 49.4, 51.1, 49.4, 47.9, 49.8, 50.9, 49.3, 51.9,
                   50.8, 49.6, 49.3, 50.6, 48.4, 50.7, 50.9, 50.6, 51.5, 52.8,
                   51.8, 51.1, 49.8, 50.2, 50.4, 51.6, 51.8, 50.9, 48.8, 51.7,
                   51.0, 50.6, 51.7, 51.5, 52.1, 51.3, 51.0, 54.0, 51.4, 52.7,
                   53.1, 54.6, 52.0, 52.0, 50.9, 52.6, 50.2, 52.6, 51.6, 51.9,
                   50.5, 50.9, 51.7, 51.4, 51.7, 50.8, 51.9, 51.8, 51.9, 53.0])

nhtemp_dates = np.arange("1912", "1972", dtype="datetime64[Y]")


"""
R dataset: Flow of the River Nile with one breakpoint: the annual flows drop
 in 1898 because the first Ashwan dam was built
"""
nile = np.array([1120, 1160, 963,  1210, 1160, 1160, 813,  1230, 1370, 1140,
                 995,  935,  1110, 994,  1020, 960,  1180, 799,  958,  1140,
                 1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774,  840,
                 874,  694,  940,  833,  701,  916,  692,  1020, 1050, 969,
                 831,  726,  456,  824,  702,  1120, 1100, 832,  764,  821,
                 768,  845,  864,  862,  698,  845,  744,  796,  1040, 759,
                 781,  865,  845,  944,  984,  897,  822,  1010, 771,  676,
                 649,  846,  812,  742,  801,  1040, 860,  874,  848,  890,
                 744,  749,  838,  1050, 918,  986,  797,  923,  975,  815,
                 1020, 906,  901,  1170, 912,  746,  919,  718,  714,  740])

# nile_dates = np.arange("1871", "1971", dtype="datetime64[Y]")
nile_dates = np.arange(1871, 1971).astype(float)


"""
A multivariate monthly time series from 1959(1) to 2001(2) with variables
from the strucchange package.
by: Achim Zeileis
"""
us_inc_exp = np.load(data_folder + "USIncExp.npy")


"""
R dataset: Time series giving the monthly totals of car drivers in Great
Britain killed or seriously injured Jan 1969 to Dec 1984. Compulsory wearing of
seat belts was introduced on 31 Jan 1983.
"""
uk_driver_deaths = np.array([1687, 1508, 1507, 1385, 1632, 1511, 1559, 1630,
                             1579, 1653, 2152, 2148, 1752, 1765, 1717, 1558,
                             1575, 1520, 1805, 1800, 1719, 2008, 2242, 2478,
                             2030, 1655, 1693, 1623, 1805, 1746, 1795, 1926,
                             1619, 1992, 2233, 2192, 2080, 1768, 1835, 1569,
                             1976, 1853, 1965, 1689, 1778, 1976, 2397, 2654,
                             2097, 1963, 1677, 1941, 2003, 1813, 2012, 1912,
                             2084, 2080, 2118, 2150, 1608, 1503, 1548, 1382,
                             1731, 1798, 1779, 1887, 2004, 2077, 2092, 2051,
                             1577, 1356, 1652, 1382, 1519, 1421, 1442, 1543,
                             1656, 1561, 1905, 2199, 1473, 1655, 1407, 1395,
                             1530, 1309, 1526, 1327, 1627, 1748, 1958, 2274,
                             1648, 1401, 1411, 1403, 1394, 1520, 1528, 1643,
                             1515, 1685, 2000, 2215, 1956, 1462, 1563, 1459,
                             1446, 1622, 1657, 1638, 1643, 1683, 2050, 2262,
                             1813, 1445, 1762, 1461, 1556, 1431, 1427, 1554,
                             1645, 1653, 2016, 2207, 1665, 1361, 1506, 1360,
                             1453, 1522, 1460, 1552, 1548, 1827, 1737, 1941,
                             1474, 1458, 1542, 1404, 1522, 1385, 1641, 1510,
                             1681, 1938, 1868, 1726, 1456, 1445, 1456, 1365,
                             1487, 1558, 1488, 1684, 1594, 1850, 1998, 2079,
                             1494, 1057, 1218, 1168, 1236, 1076, 1174, 1139,
                             1427, 1487, 1483, 1513, 1357, 1165, 1282, 1110,
                             1297, 1185, 1222, 1284, 1444, 1575, 1737, 1763])

uk_driver_deaths_dates = np.arange("1969-01", "1985-01", dtype="datetime64[M]")


"""
NDVI time series, simulated by extracting key characteristics from MODIS 16-day
NDVI time series.
"""
ndvi = np.load(data_folder + "ndvi.npy")
ndvi_freq = 24
ndvi_dates = r_style_interval((1982, 1), (2011, 24), ndvi_freq).reshape(ndvi.shape[0], 1)


"""
SIMTS dataset
"""
simts_freq = 23
simts = np.load(data_folder + "simts.npy")
simts_sum = np.sum(simts, axis=2).reshape(simts.shape[1])
simts_dates = r_style_interval((2000, 4), (2008, 18), simts_freq).reshape(simts.shape[1], 1)


"""
harvest dataset
"""
harvest_freq = 23
harvest = np.load(data_folder + "harvest.npy")
harvest_dates = r_style_interval((2000, 4), (2008, 18), harvest_freq).reshape(harvest.shape[0], 1)


# """
# Test with breakpoints in both seasonal and trend
# """
# _both_dates = r_style_interval((1990, 1), (1999, 24), ndvi_freq)
# _both_n = _both_dates.shape[0]
# both_freq = 24
# _both_x = np.arange(_both_n)
# _both_harm = (np.sin(_both_x * 0.5))
# _both_harm[150:] *= 3
# _both_trend = 0.02 * _both_x
# # _both_trend[100:] += 5

# both_dates = _both_dates.reshape(_both_n, 1)
# # both = _both_trend + _both_harm
# both = _both_harm

if __name__ == "__main__":
    print(ndvi)
    print(simts_sum.shape)
    print(simts_dates.shape)
    # plt.plot(both_dates, both)
    # plt.show()
