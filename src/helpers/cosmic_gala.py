from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import independent
from cosmic.evolve import Evolve

import numpy as np
import gala

import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const

from schwimmbad import MultiPool

import matplotlib.pyplot as plt

from galaxy import draw_lookback_times, draw_radii, draw_heights, R_exp


def final_systemic_velocity(v_sys_init, delta_v_sys_xyz, m_1, m_2, a):
    """Calculate the final systemic velocity from a combination of the initial systemic velocity, natal kick,
    Blauuw kick and orbital motion.

    Parameters
    ----------
    v_sys_init : `float array`
        Initial system velocity in Galactocentric (v_R, v_T, v_z) coordinates
    delta_v_sys_xyz : `float array`
        Change in systemic velocity due to natal and Blauuw kicks in BSE (v_x, v_y, v_z) frame (see Fig A1 of
        Hurley+02)
    m_1 : `float`
        Primary mass
    m_2 : `float`
        Secondary Mass
    a : `float`
        Separation

    Returns
    -------
    v_sys_final : `float array`
        Final systemic velocity
    """
    # calculate the orbital velocity ASSUMING A CIRCULAR ORBIT
    v_orb = np.sqrt(const.G * (m_1 + m_2) / a)

    # adjust change in velocity by orbital motion of supernova star
    delta_v_sys_xyz -= v_orb
    delta_v_sys_xyz = delta_v_sys_xyz.to(u.km/u.s).value

    # orbital phase angle and inclination to Galactic plane
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    # rotate (v_x, v_y, v_z) into Galactocentric (v_R, v_T, v_z')
    delta_v_sys_RTz = [delta_v_sys_xyz[0] * np.cos(theta)
                       - delta_v_sys_xyz[1] * np.sin(theta) * np.cos(phi)
                       + delta_v_sys_xyz[2] * np.sin(theta) * np.sin(phi),
                       delta_v_sys_xyz[0] * np.sin(theta)
                       + delta_v_sys_xyz[1] * np.cos(theta) * np.cos(phi)
                       - delta_v_sys_xyz[2] * np.cos(theta) * np.sin(phi),
                       delta_v_sys_xyz[1] * np.sin(phi)
                       + delta_v_sys_xyz[2] * np.cos(phi)] * u.km / u.s

    return v_sys_init + delta_v_sys_RTz
