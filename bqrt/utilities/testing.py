def slope_intercept(x1, y1, x2, y2):
    """ Testing liear inpterpolation

    Parameters
    ----------
    x1 : float
        x1
    y1 : float
        y1
    x2 : float
        x2
    y2 : float
        y2

    Returns
    -------
    float
        return linear interpolation slope and intercept
    """
    slope = (y2 - y1) / (x2 - x1)
    intercept = (x1 * y2 - x2 * y1) / (x1 - x2)
    return slope, intercept
