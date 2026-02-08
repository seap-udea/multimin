import multimin as mn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NEA Data
df_neas=pd.read_json(mn.Util.get_data("nea_data.json.gz"))

# Let's filter 10000 asteroids
df_neas=df_neas.sample(10000)

# Let's select the columns we want to fit
df_neas["q"]=df_neas["a"]*(1-df_neas["e"])
data_neas=np.array(df_neas[["q","e","i","Node","Peri","M"]])

data_neas_qei = np.array(df_neas[["q","e","i"]])

fit_qei = mn.FitCMND(ngauss=1, nvars=3, domain=[[0, 1.3], [0, 1.0], [0, 180]])
# 1D arrays: same initial values for all components (here ngauss=1, so equivalent to [[...]])
fit_qei.set_initial_params(
    mus=[1.0, 0.7, 20.0],
    sigmas=[0.1, 0.3, 5.0],
    rhos=[-0.5, 0.0, 0.0],
)
# normalize=True: fit in [0,1] per variable for better conditioning and more stable minima
fit_qei.fit_data(data_neas_qei, advance=True, normalize=True)
print(f"-log(L)/N = {fit_qei.solution.fun / len(data_neas_qei)}")

props=["q","e","i"]
hargs=dict(bins=30,cmap='YlGn')
sargs=dict(s=0.5,edgecolor='None',color='r')
G=fit_qei.plot_fit(
    props=props,
    hargs=hargs,
    sargs=sargs,
    figsize=3
)
plt.savefig(f'gallery/test_fit_result_qei_1gauss.png')