import numpy as np

def htanh(x: np.ndarray[:], a: (int | float)) -> np.ndarray[:]:
    """
    Compute a hard tanh activation on the input x.

    This function applies a piecewise transformation to x:
      - When |x| ≤ a, it returns x/a.
      - When |x| > a, it returns 1 if x is positive, and -1 otherwise.
      - For a = 0, the function directly returns the sign of x.

    Parameters:
      x : array-like or scalar
          The input value(s) to be transformed.
      a : float
          The threshold parameter (with |a| ≤ 1) that defines the linear region.

    Returns:
      Transformed value(s) following the hard tanh definition applied element-wise.

    Raises:
      AssertionError: If the absolute value of a is greater than 1.
    """

    assert abs(a) <= 1

    if a == 0:
        # purely sign‐based when a==0
        return np.sign(x)
    # scale into [−1,1] and clip
    return np.clip(x / a, -1, 1)