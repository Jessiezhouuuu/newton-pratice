import warnings
import numpy as np
from scipy.optimize import approx_fprime

def optimize(start, fun, max_iter=10000, tol=1e-7):
    """
    Run Newton method to minimise a function.

    Parameters
    ----------
    start :
        initial value

    fun :
        your target function

    max_iter :
        maximum iterations

    tol :
        tolerance level

    Returns
    -------
    x :
        minimizer found by Newton method
    """
    if not callable(fun):
       raise TypeError(f"Argument is not a function, it is of type {type(fun)}")

    # if x > 3:
    #    warnings.warn(f"{x} is greater than 3.", UserWarning)

    ## calculate derivative
    def first_derivative(x, fun, h=1e-5):
        return (fun(x + h) - fun(x - h)) / (2 * h)

    def second_derivate(x, fun, h=1e-5):
        return (first_derivative(x + h, fun) - first_derivative(x - h, fun)) / (2 * h)

    x = float(start)

    for i in range(max_iter):
        f_prime = first_derivative(x, fun)
        f_second_prime = second_derivate(x, fun)
        try:
            new_x = x - f_prime / f_second_prime
        except ZeroDivisionError:
            print('The second derivative equals to zero!')
            return 
        change = abs(x - new_x)
        if change < tol:
            break
        x = new_x
        if x > 1e7:
            raise RuntimeError(f"At iteration {i}, optimization appears to be diverging")

    return {'x':x,'value':fun(x)}

def numerical_hessian(fun, x, epsilon=1e-5):
	n = x.size
	hess = np.zeros((n, n))
	fx = fun(x)
	for i in range(n):
		x_i1 = np.array(x, dtype=float)
		x_i1[i] += epsilon
		grad1 = approx_fprime(x_i1, fun, epsilon)
		x_i2 = np.array(x, dtype=float)
		x_i2[i] -= epsilon
		grad2 = approx_fprime(x_i2, fun, epsilon)
		hess[:, i] = (grad1 - grad2) / (2 * epsilon)
	return hess

def multi_optimize(start, fun, max_iter=10000, tol=1e-7):
	x = start
	iter = 0
	while iter < max_iter:
		grad = approx_fprime(x, fun, 1e-8)
		hess = numerical_hessian(fun, x)
		try:
			step = np.linalg.solve(hess, grad)
		except np.linalg.LinAlgError:
			print("Hessian is not invertible.")
			break
		x_new = x - step
		if np.linalg.norm(x_new - x) < tol:
			print(f"Converged after {iter+1} iterations.")
			x = x_new
			break
		x = x_new
		iter += 1
	return {"Minimum": x, "Function Value": fun(x), "Iterations": iter}
