# TODO: Jax this code and make it extremely dope. 
# This is the model of the front pupil. 
def pupil(ys, xs, pix_rad, pdiam, odiam=0.0,
                     beta=45.0, thick=0.25, offset=0.0,
                     spiders=True, split=False, between_pix=True):
    beta = beta * dtor  # converted to radians
    ro = odiam / pdiam
    yy, xx = _xyic(ys, xs, between_pix=between_pix)
    mydist = np.hypot(yy, xx)

    thick *= pix_rad / pdiam
    offset *= pix_rad / pdiam

    x0 = thick/(2 * np.sin(beta)) + offset
    y0 = thick/(2 * np.cos(beta)) - offset * np.tan(beta)

    if spiders:
        # quadrants left - right
        a = ((xx >= x0) * (np.abs(np.arctan(yy/(xx-x0+1e-8))) < beta))
        b = ((xx <= -x0) * (np.abs(np.arctan(yy/(xx+x0+1e-8))) < beta))
        # quadrants up - down
        c = ((yy >= 0.0) * (np.abs(np.arctan((yy-y0)/(xx+1e-8))) > beta))
        d = ((yy < 0.0) * (np.abs(np.arctan((yy+y0)/(xx+1e-8))) > beta))

    # pupil outer and inner edge
    e = (mydist <= np.round(pix_rad))
    if odiam > 1e-3:  # threshold at 1 mm
        e *= (mydist > np.round(ro * pix_rad))

    if split:
        res = np.array([a*e, b*e, c*e, d*e])
        return(res)

    if spiders:
        return((a+b+c+d)*e)
    else:
        return(e)
