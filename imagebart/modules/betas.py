import numpy as np
import pickle


def find_roots(x,y):
    # https://stackoverflow.com/questions/46909373/how-to-find-the-exact-intersection-of-a-curve-as-np-array-with-y-0/46911822#46911822
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


def load_betas(path, dist, metric, n_steps):
    with open(path, "rb") as f:
        data = pickle.load(f)
    alphacum = np.array(data[dist]["alphacum"])
    values = np.array(data[dist][metric])[:,0]

    equi_alphacum = list()
    for val in np.linspace(values.min(), values.max(), n_steps):
        equi_alphacum.append(find_roots(alphacum, values-val)[0])
    equi_alphacum = np.array(equi_alphacum)
    beta_1toT = 1-equi_alphacum[1:]/equi_alphacum[:-1]
    beta_0toT = np.concatenate((np.array([0.0]), beta_1toT))
    return beta_0toT.astype(np.float32)


if __name__ == "__main__":
    import sys
    betas = load_betas(sys.argv[1], "bernoulli", "FID", int(sys.argv[2]))
    print(betas)
    print(np.cumprod(1-betas, 0))
