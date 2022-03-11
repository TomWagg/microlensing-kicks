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


def evolve_binaries_in_galaxy(bpp, kick_info, galaxy_model=None,
                              galactic_potential=gala.potential.MilkyWayPotential(),
                              max_ev_time=13.7 * u.Gyr, dispersion=5 * u.km / u.s):
    # work out how many binaries we are going to evolve
    bin_nums = bpp["bin_num"].unique()
    n_bin = len(bin_nums)

    vel_units = u.km / u.s

    # draw random positions and birth times in the galaxy
    # TODO: actually make this change based on the `galaxy model`
    lookback_time = draw_lookback_times(n_bin, tm=12 * u.Gyr, tsfr=6.8 * u.Gyr, component="low_alpha_disc")
    scale_length = R_exp(lookback_time, alpha=0.4)
    rho = draw_radii(n_bin, R_0=scale_length)
    scale_height = 0.3 * u.kpc
    z = draw_heights(n_bin, z_d=scale_height)
    phi = np.random.uniform(0, 2 * np.pi, size=n_bin) * u.rad

    # calculate the Galactic circular velocity at the given positions
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    v_circ = gala.potential.MilkyWayPotential().circular_velocity(q=[x, y, z]).to(vel_units)

    # add some velocity dispersion
    v_R, v_T, v_z = np.random.normal([np.zeros_like(v_circ), v_circ, np.zeros_like(v_circ)],
                                     dispersion.to(vel_units) / np.sqrt(3), size=(3, n_bin))
    v_R, v_T, v_z = v_R * vel_units, v_T * vel_units, v_z * vel_units

    # turn the drawn coordinates into an astropy representation
    rep = coords.CylindricalRepresentation(rho, phi, z)

    # create differentials based on the velocities (dimensionless angles allows radians conversion)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        dif = coords.CylindricalDifferential(v_R, (v_T / rho).to(u.rad / u.Gyr), v_z)

    # combine the representation and differentials into a Gala PhaseSpacePosition
    w0s = gala.dynamics.PhaseSpacePosition(rep.with_differentials(dif))

    # evolve the orbits from birth until present day
    # TODO: make this actually account for kicks haha (baby steps)
    orbits = []
    for bin_num in bin_nums:
        orbits.append(galactic_potential.integrate_orbit(w0s[bin_num], t1=lookback_time[bin_num],
                                                         t2=max_ev_time, dt=1 * u.Myr))

    return orbits
