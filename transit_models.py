"""
Transit model helper functions
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class StellarParams:
    """Host star properties"""
    R_star: float
    M_star: float
    u1: float
    u2: float


def expected_duration(P, stellar, impact):
    """
    Calculate transit duration from period using Kepler's 3rd law
    
    Args:
        P: orbital period (days)
        stellar: StellarParams object
        impact: impact parameter (0-1)
    
    Returns:
        duration in days
    """
    a_AU = (stellar.M_star * (P / 365.25)**2)**(1/3)
    a_Rsun = a_AU * 215.032
    
    if impact >= 1.0:
        return 0.0
    
    duration = (P / np.pi) * np.arcsin(
        stellar.R_star / a_Rsun * np.sqrt(1 - impact**2)
    )
    
    return duration


def limb_darkened_transit(time, t0, RpRs, P, impact, stellar, model='batman'):
    """
    Transit model with quadratic limb darkening
    
    Supports batman (Mandel & Agol), pytransit (Parviainen), or ellc
    
    Args:
        time: observation times
        t0: mid-transit time
        RpRs: planet-star radius ratio
        P: period
        impact: impact parameter
        stellar: StellarParams object
        model: 'batman', 'pytransit', or 'ellc'
    """
    # Semi-major axis in stellar radii (from Kepler's 3rd law)
    a_AU = (stellar.M_star * (P / 365.25)**2)**(1/3)
    a_over_rs = a_AU * 215.032 / stellar.R_star
    
    if model == 'ellc':
        import ellc
        
        r_1 = 1.0 / a_over_rs
        r_2 = RpRs / a_over_rs
        inc = np.degrees(np.arccos(impact / a_over_rs))
        
        flux = ellc.lc(
            t_obs=time,
            radius_1=r_1,
            radius_2=r_2,
            sbratio=0.0,
            incl=inc,
            t_zero=t0,
            period=P,
            a=a_over_rs,
            q=0.001,
            ldc_1=[stellar.u1, stellar.u2],
            ld_1='quad',
            shape_1='sphere',
            shape_2='sphere'
        )
        return flux
    
    elif model == 'pytransit':
        from pytransit import QuadraticModel
        
        tm = QuadraticModel()
        tm.set_data(time)
        
        k = RpRs
        inc = np.degrees(np.arccos(impact / a_over_rs))
        
        flux = tm.evaluate(k, [stellar.u1, stellar.u2], t0, P, a_over_rs, inc)
        return flux
    
    else:
        import batman
        
        params = batman.TransitParams()
        params.t0 = t0
        params.per = P
        params.rp = RpRs
        params.a = a_over_rs
        params.inc = np.degrees(np.arccos(impact / a_over_rs))
        params.ecc = 0.0
        params.w = 90.0
        params.u = [stellar.u1, stellar.u2]
        params.limb_dark = "quadratic"
        
        m = batman.TransitModel(params, time)
        flux = m.light_curve(params)
        
        return flux


def simple_transit(time, t0, RpRs, P, impact, stellar):
    """
    Simplified transit model (faster for grid search)
    Trapezoidal with stellar-constrained duration
    
    Args:
        RpRs: planet-star radius ratio (not depth)
    """
    flux = np.ones_like(time)
    
    # Get duration from stellar params
    duration = expected_duration(P, stellar, impact)
    
    if duration == 0:
        return flux
    
    # Calculate depth from radius ratio
    depth = RpRs ** 2
    
    # Find which transit each point belongs to
    transit_num = np.round((time - t0) / P)
    transit_times = t0 + transit_num * P
    dt = time - transit_times
    
    # Trapezoidal shape
    ingress_frac = 0.2
    ingress = ingress_frac * duration / 2
    
    in_transit = np.abs(dt) < duration / 2
    in_ingress = (dt > -duration/2) & (dt < -duration/2 + ingress)
    in_egress = (dt > duration/2 - ingress) & (dt < duration/2)
    
    # Scale depth by impact (grazing transits are shallower)
    impact_factor = 1.0 - 0.3 * impact  # Rough approximation
    effective_depth = depth * impact_factor
    
    flux[in_transit] = 1 - effective_depth
    flux[in_ingress] = 1 - effective_depth * (dt[in_ingress] + duration/2) / ingress
    flux[in_egress] = 1 - effective_depth * (duration/2 - dt[in_egress]) / ingress
    
    return flux
