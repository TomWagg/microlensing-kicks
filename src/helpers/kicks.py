import gala as ga
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from scipy.stats import maxwell


def integrate_orbits_with_kicks(potential, w0, kicks=None, kick_times=None, maxwell_sigma=265 * u.km / u.s,
                                same_angle=False, **integrate_kwargs):
    """Integrate PhaseSpacePosition in a potential with kicks that occur at certain times

    Parameters
    ----------
    potential : `ga.potential.PotentialBase`
        Potential in which you which to integrate the orbits
    w0 : `ga.dynamics.PhaseSpacePosition`
        Initial phase space position
    kicks : `list`, optional
        List of None, or list of kick magnitudes or list of tuples with kick magnitudes and angles,
        by default None
    kick_times : `list`, optional
        Times at which kicks occur, by default None
    maxwell_sigma : `float`
        Sigma to use for the maxwellian for kick magnitudes
    same_angle : `boolean`
        Whether to use the same random kick angle for each individual orbit if several are provided

    Returns
    -------
    full_orbits : `ga.orbit.Orbit`
        Orbits that have been integrated
    """
    # if there are no kicks then just integrate the whole thing
    if kicks is None and kick_times is None:
        return potential.integrate_orbit(w0, **integrate_kwargs)

    # otherwise make sure that both are there
    elif kicks is None or kick_times is None:
        raise ValueError("Both kicks and times must be specified")

    # then integrate using the kicks
    else:
        # work out what the timesteps would be without kicks
        timesteps = ga.integrate.parse_time_specification(units=[u.s], **integrate_kwargs) * u.s

        # start the cursor at the smallest timestep
        time_cursor = timesteps[0]
        current_w0 = w0

        # keep track of the orbit data throughout
        data = []
        for kick, kick_time in zip(kicks, kick_times):
            # find the timesteps that occur before the kick
            matching_timesteps = timesteps[np.logical_and(timesteps >= time_cursor, timesteps < kick_time)]

            # integrate the orbit over these timesteps
            orbits = potential.integrate_orbit(current_w0, t=matching_timesteps)

            # save the orbit data
            data.append(orbits.data)

            # adjust the time
            time_cursor = kick_time

            # get new PhaseSpacePosition(s)
            current_w0 = orbits[-1]

            if isinstance(kick, tuple):
                magnitude, phi, theta = kick
            else:
                # if there's only one orbit
                if current_w0.shape == ():
                    magnitude = kick if kick is not None\
                        else maxwell(maxwell_sigma).rvs() * maxwell_sigma.unit
                    phi = np.random.uniform(0, 2 * np.pi)
                    theta = np.random.uniform(-np.pi / 2, np.pi / 2)
                else:
                    magnitude = kick if kick is not None else\
                        maxwell(maxwell_sigma).rvs(current_w0.shape[0]) * maxwell_sigma.unit

                    if same_angle:
                        phi_0 = np.random.uniform(0, 2 * np.pi)
                        theta_0 = np.random.uniform(-np.pi / 2, np.pi / 2)
                        phi = np.repeat(phi_0, repeats=current_w0.shape[0])
                        theta = np.repeat(theta_0, repeats=current_w0.shape[0])
                    else:
                        phi = np.random.uniform(0, 2 * np.pi, size=current_w0.shape[0])
                        theta = np.random.uniform(-np.pi / 2, np.pi / 2, size=current_w0.shape[0])

            d_x = magnitude * np.cos(phi) * np.sin(theta)
            d_y = magnitude * np.sin(phi) * np.sin(theta)
            d_z = magnitude * np.cos(theta)

            kick_differential = coord.CartesianDifferential(d_x, d_y, d_z)

            current_w0 = ga.dynamics.PhaseSpacePosition(pos=current_w0.pos,
                                                        vel=current_w0.vel + kick_differential,
                                                        frame=current_w0.frame)

        if time_cursor < timesteps[-1]:
            matching_timesteps = timesteps[timesteps >= time_cursor]
            orbits = potential.integrate_orbit(current_w0, t=matching_timesteps)
            data.append(orbits.data)

        data = coord.concatenate_representations(data)
        full_orbits = ga.dynamics.orbit.Orbit(pos=data.without_differentials(),
                                              vel=data.differentials["s"],
                                              t=timesteps.to(u.Myr))

        return full_orbits




def one_kicked_orbit(w0, kick_time=None, sigma = 0*u.km/u.s,
                     kick=None, no_kicks = None,
                     potential=ga.potential.MilkyWayPotential(),
                     **integrate_kwargs):
    """ calculates one orbit adding a random kick

    Parameters
    ----------
    w0 : `ga.dynamics.PhaseSpacePosition`
        Initial phase space position
    kick_time: `float`
       Times at which kicks occur
    sigma : `float`
        Sigma to use for the maxwellian for kick magnitudes in km/s
    kick : `float`, optional
        kick magnitude or tuple with kick magnitude and angles phi and theta,
        by default None. If not None will over-ride random drawing of kicks
    no_kicks : `ga.orbit.Orbit`
       unkicked orbit, to avoid recomputing it
    potential : `ga.potential.PotentialBase`
        Potential in which you which to integrate the orbits,
        by default MilkyWayPotential in gala.potential

    Returns
    -------
    full_orbits : `ga.orbit.Orbit`
        Orbits that have been integrated
    kick : tuple
    (magnitude, theta, phi)
    """
    if ((kick is not None) and (sigma >=0)):
        raise ValueError("Specify either kick list or sigma for Maxwellian")
    elif ((kick is not None) or (sigma>=0)) and (kick_time is  None):
        raise ValueError("Must specify time to kick")
    elif (kick is None) and (sigma <= 0) and (kick_time is not None):
        print("no kicks!")
        if no_kicks:
            return no_kicks, (0,0,0)
        else:
            return potential.integrate_orbit(w0, **integrate_kwargs), (0,0,0)
    else:
        # work out what the timesteps would be without kicks
        if no_kicks:
            timesteps = no_kicks.t
        else:
            timesteps = ga.integrate.parse_time_specification(units=[u.s], **integrate_kwargs) * u.s
        # start the cursor at the smallest timestep
        time_cursor = timesteps[0]
        current_w0 = w0
        # keep track of the orbit data throughout
        data = []

        # find the timesteps that occur before the kick
        matching_timesteps = timesteps[np.logical_and(timesteps >= time_cursor, timesteps < kick_time)]

        # integrate the orbit over these timesteps
        orbits = potential.integrate_orbit(current_w0, t=matching_timesteps)

        # save the orbit data
        data.append(orbits.data)

        # adjust the time
        time_cursor = kick_time

        # get new PhaseSpacePosition(s)
        current_w0 = orbits[-1]

        # Now kick the orbit
        if kick:
            print("Using the kick given", kick)
            if isinstance(kick, tuple):
                magnitude, phi, theta = kick
            else:
                magnitude = kick
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        else:
            # draw a kick
            magnitude = maxwell(sigma).rvs() * sigma.unit
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)

        d_x = magnitude * np.cos(phi) * np.sin(theta)
        d_y = magnitude * np.sin(phi) * np.sin(theta)
        d_z = magnitude * np.cos(theta)

        kick_differential = coord.CartesianDifferential(d_x, d_y, d_z)

        current_w0 = ga.dynamics.PhaseSpacePosition(pos=current_w0.pos,
                                                    vel=current_w0.vel + kick_differential,
                                                    frame=current_w0.frame)

        if time_cursor < timesteps[-1]:
            matching_timesteps = timesteps[timesteps >= time_cursor]
            orbits = potential.integrate_orbit(current_w0, t=matching_timesteps)
            data.append(orbits.data)

        data = coord.concatenate_representations(data)
        full_orbits = ga.dynamics.orbit.Orbit(pos=data.without_differentials(),
                                              vel=data.differentials["s"],
                                              t=timesteps.to(u.Myr))

        return full_orbits, (magnitude, theta, phi)
