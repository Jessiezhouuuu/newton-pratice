import warnings
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
