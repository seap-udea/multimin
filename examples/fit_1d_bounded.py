import multimin as mn
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

CMND_1d = mn.ComposedMultiVariateNormal(
    mus=[0.2, 0.8],
    weights=[0.5, 0.5],
    Sigmas=[0.02, 0.02],
    domain=[[0, 1]]  # variable 0 bounded to [0, 1]
    #domain=[None]  # variable 0 bounded to [0, 1]
)

np.random.seed(42)
data_1d = CMND_1d.rvs(5000)

F_1d = mn.FitCMND(ngauss=2, nvars=1, domain=[[0, 1]])
F_1d.fit_data(data_1d, advance=True)

tabla1 = CMND_1d.tabulate(sort_by="distance")
table2 = F_1d.cmnd.tabulate(sort_by="distance")

print(tabla1)
print(table2)