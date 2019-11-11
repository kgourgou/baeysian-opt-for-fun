import scipy as sc

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as gp
from scipy.optimize import minimize_scalar

def expected_improvement(x, current_max, model):
    """
    Calculates the expected improvement function, 
    an acquisition function used in Bayesian optimization.

    Args
    ====
    x: float, where to evaluate the target function next.  
    current_max: float, should be the most recent max value 
    of the target function. 
    model: sklearn object, should be a GP that is trained on
    the current data.

    """
    y, std = model.predict(sc.array([[x]]), return_std=True)
    Dm = y-current_max
    gamma = Dm / std

    exp_imp = max([Dm,0])
    exp_imp += std*norm.pdf(gamma)-abs(Dm)*norm.cdf(gamma)
    exp_imp = exp_imp[0]
    return exp_imp


def bayes_opt(known_points, fun, n_iter, bounds=(-10,10)):
    """
    Bayesian optimization with the simplest acquisition function,
    "expected_improvement". Goal is to find the global maximum of 
    a target function, fun.

    Args
    ====
    known_points: 1D array, contains the points where fun has been evaluated.
    fun: the target function. 
    n_iter: number of iterations through Bayesian optimization. 
    bounds: tuple, the bounds to be used during optimization. Optional. 
    """

    kp_list = list(known_points)
    y_array = fun(known_points)
    y = list(y_array)

    model = gp().fit(known_points.reshape(-1,1),y_array)

    known_point_pred = model.predict(known_points.reshape(-1,1))
    running_max = max(known_point_pred)

    for i in range(n_iter):
        g = lambda x: -expected_improvement(x,running_max, model)

        opt = minimize_scalar(g,
                              bounds=bounds,
                              method='Bounded')
        xnew = opt['x']
        ynew = fun(xnew)
        running_max = max([running_max, ynew])

        kp_list.append(xnew)
        y.append(running_max)

        known_points = sc.array(kp_list)
        y_array = fun(known_points)

        # refit to include the new points - although there's probably a more efficient
        # way than this. 
        model = gp().fit(known_points.reshape(-1,1), y_array)

    return running_max, known_points, y, model
