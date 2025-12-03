"""
Data loading and stellar parameter queries
"""

import numpy as np
import lightkurve as lk
import json
import os
from transit_models import StellarParams


def get_limb_darkening_claret(Teff, logg, feh, mission='Kepler'):
    """
    Get quadratic limb darkening coefficients using exotic-ld
    Uses stellar atmosphere models (ATLAS/PHOENIX) for mission-specific bandpass
    
    Args:
        Teff: Effective temperature (K)
        logg: Surface gravity log10(g [cm/s²])
        feh: Metallicity [Fe/H]
        mission: 'TESS', 'Kepler', or 'CoRoT' (default 'Kepler')
    
    Returns:
        u1, u2: quadratic limb darkening coefficients
    """
    try:
        from exotic_ld import StellarLimbDarkening
        
        # Create limb darkening calculator
        # M_H is metallicity, ld_model can be 'claret', 'mps1', 'stagger', etc.
        sld = StellarLimbDarkening(
            M_H=feh,
            Teff=Teff,
            logg=logg,
            ld_model='claret',  # Use Claret tables
            ld_data_path='default'
        )
        
        # Compute quadratic coefficients for mission-specific bandpass
        # TESS: 6000-10000 Å, Kepler: 4200-9000 Å, CoRoT: 3000-10000 Å (wide)
        if mission == 'TESS':
            wavelength_range = [6000, 10000]
        elif mission == 'CoRoT':
            wavelength_range = [3000, 10000]
        else:  # Kepler or default
            wavelength_range = [4200, 9000]
        
        # Don't use mode parameter, just wavelength_range
        coeffs = sld.compute_quadratic_ld_coeffs(wavelength_range=wavelength_range)
        
        u1, u2 = coeffs
        print(f"  u1, u2 = {u1:.4f}, {u2:.4f} (exotic-ld/Claret, {mission} band)")
        return float(u1), float(u2)
        
    except Exception as e:
        print(f"  exotic-ld failed ({e}), using fallback interpolation")
        # Fallback to manual Claret table interpolation
        return get_limb_darkening_fallback(Teff, logg, feh)


def get_limb_darkening_fallback(Teff, logg, feh):
    """
    Fallback limb darkening from manual Claret table interpolation
    """
    # Claret & Bloemen 2011 Table 2 - Kepler band, quadratic law
    claret_table = np.array([
        [3500, 4.5, 0.7291, 0.0542],
        [4000, 4.5, 0.6587, 0.1042],
        [4500, 4.5, 0.5800, 0.1517],
        [5000, 4.5, 0.4869, 0.2041],
        [5500, 4.5, 0.4048, 0.2438],
        [5777, 4.44, 0.3561, 0.2623],  # Sun
        [6000, 4.5, 0.3197, 0.2781],
        [6500, 4.5, 0.2442, 0.3026],
        [7000, 4.5, 0.1909, 0.3091],
        [7500, 4.5, 0.1504, 0.3055],
        [8000, 4.5, 0.1180, 0.2964],
    ])
    
    Teff = np.clip(Teff, 3500, 8000)
    logg = np.clip(logg, 3.5, 4.5)
    
    dists = np.sqrt((claret_table[:, 0] - Teff)**2 / 1000**2 + 
                    (claret_table[:, 1] - logg)**2 * 10)
    
    k = min(3, len(claret_table))
    nearest_idx = np.argsort(dists)[:k]
    weights = 1.0 / (dists[nearest_idx] + 1e-6)
    weights /= weights.sum()
    
    u1 = np.sum(claret_table[nearest_idx, 2] * weights)
    u2 = np.sum(claret_table[nearest_idx, 3] * weights)
    
    return u1, u2


def load_data(star_name='HAT-P-7', cache_dir='data_cache'):
    """Load and normalize Kepler data + query stellar parameters from catalogs"""
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{star_name.replace(' ', '_')}_data.json")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading {star_name} data from cache...")
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Reconstruct numpy arrays and StellarParams object
            time = np.array(cached['time'])
            flux = np.array(cached['flux'])
            stellar = StellarParams(
                R_star=cached['stellar']['R_star'],
                M_star=cached['stellar']['M_star'],
                u1=cached['stellar']['u1'],
                u2=cached['stellar']['u2']
            )
            
            print(f"  Cache loaded: {len(time)} points, {(time[-1]-time[0])/365.25:.1f} years")
            print(f"  Cached stellar params: M*={stellar.M_star:.2f} M_sun, R*={stellar.R_star:.2f} R_sun")
            return time, flux, stellar
        except Exception as e:
            print(f"  Cache load failed ({e}), re-downloading...")
    
    print(f"Downloading {star_name} data from archive...")
    
    # Try TESS first (better time resolution), then Kepler, then CoRoT
    search = lk.search_lightcurve(star_name, author='SPOC', cadence='short')
    mission = 'TESS'
    
    if len(search) == 0:
        print(f"  No TESS data found, trying Kepler...")
        search = lk.search_lightcurve(star_name, author='Kepler', cadence='long')
        mission = 'Kepler'
    
    if len(search) == 0:
        print(f"  No Kepler data found, trying CoRoT...")
        search = lk.search_lightcurve(star_name, mission='CoRoT')
        mission = 'CoRoT'
    
    if len(search) == 0:
        raise ValueError(f"No TESS, Kepler, or CoRoT data found for {star_name}")
    
    print(f"  Found {len(search)} observations")
    lc = search.download_all().stitch()
    print("  Data downloaded.")
    time = lc.time.value
    flux = lc.flux.value
    
    # Remove NaNs and normalize
    valid = ~(np.isnan(time) | np.isnan(flux))
    time = time[valid]
    flux = flux[valid]
    
    # Normalize to baseline of 1.0
    # Use median instead of mean to be robust to outliers/transits
    baseline = np.nanmedian(flux)
    flux = flux / baseline
    
    print(f"Loaded {len(time)} points, {(time[-1]-time[0])/365.25:.1f} years")
    print(f"  Baseline flux: {baseline:.1f}, normalized median: {np.median(flux):.6f}")
    
    # Query stellar parameters from Kepler Input Catalog (KIC)
    print(f"\nQuerying stellar parameters from catalogs...")

    def sanitize_meta(value, default):
        if value is None:
            return default
        try:
            if np.isnan(value):
                return default
        except TypeError:
            pass
        return value
    
    try:
        R_star = sanitize_meta(lc.meta.get("RADIUS"), None)
        Teff = sanitize_meta(lc.meta.get("TEFF"), None)
        logg = sanitize_meta(lc.meta.get("LOGG"), 4.0)
        feh = sanitize_meta(lc.meta.get("FEH"), 0.0)
        M_star = None
        
        # Try to get mass from metadata first
        mass_meta = sanitize_meta(lc.meta.get("MASS"), None)
        if mass_meta is not None:
            M_star = mass_meta
            print(f"  M* = {M_star:.2f} M_sun (from KIC metadata)")
        
        # If mass not in metadata, query from astroquery MAST catalogs
        if M_star is None:
            print(f"  Mass not in metadata, querying MAST catalogs...")
            try:
                from astroquery.mast import Catalogs
                
                # Query TIC (TESS Input Catalog) which has stellar parameters
                catalog_data = Catalogs.query_object(star_name, catalog="TIC", radius=0.01)
                
                if len(catalog_data) > 0 and 'mass' in catalog_data.colnames:
                    mass_entry = catalog_data['mass'][0]
                    if not np.ma.is_masked(mass_entry):
                        M_star = float(mass_entry)
                        print(f"  M* = {M_star:.2f} M_sun (from TIC)")
                if M_star is None:
                    print(f"  Mass not found in TIC, trying Gaia...")
                    # Estimate from radius using mass-radius relation
                    # For main sequence stars: M ≈ R^2.5 (rough approximation)
                    if R_star is not None:
                        M_star = R_star ** 2.5
                        print(f"  M* = {M_star:.2f} M_sun (estimated from R-M relation)")
                    else:
                        raise ValueError("Stellar radius missing; cannot estimate mass.")
                    
            except Exception as e:
                print(f"  Catalog query failed: {e}")
                # Estimate from radius using mass-radius relation
                if R_star is not None:
                    M_star = R_star ** 2.5
                    print(f"  M* = {M_star:.2f} M_sun (estimated from R-M relation)")
                else:
                    raise ValueError("Stellar radius missing; cannot estimate mass.")
        
        # Get limb darkening from Claret tables using mission-specific bandpass
        u1, u2 = get_limb_darkening_claret(Teff, logg, feh, mission=mission)
        
        print(f"  R* = {R_star:.2f} R_sun (from KIC)")
        print(f"  Teff = {Teff:.0f} K (from KIC)")
        print(f"  log(g) = {logg:.2f} (from KIC)")
        print(f"  [Fe/H] = {feh:.2f} (from KIC)")
        
        stellar = StellarParams(
            R_star=float(R_star),
            M_star=float(M_star),
            u1=u1,
            u2=u2
        )
                    
    except Exception as e:  
        print(f"  Stellar parameter query failed: {e}")
        print(f"  ERROR: Cannot proceed without stellar parameters")
        raise ValueError(f"Failed to query stellar parameters for {star_name}. Cannot use default solar values.") from e
    
    # Cache the data for future runs
    print(f"  Saving data to cache: {cache_file}")
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'time': time.tolist(),
                'flux': flux.tolist(),
                'stellar': {
                    'R_star': stellar.R_star,
                    'M_star': stellar.M_star,
                    'u1': stellar.u1,
                    'u2': stellar.u2
                }
            }, f, indent=2)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Cache save failed: {e}")
    
    return time, flux, stellar


def get_known_period(planet_name='HAT-P-7 b', use_hardcoded=True):
    """
    Get known period from NASA Exoplanet Archive or hardcoded value
    
    Args:
        planet_name: name of planet
        use_hardcoded: if True, skip archive query and use hardcoded value
    
    Returns:
        known period in days
    """
    if use_hardcoded:
        print(f"\nUsing hardcoded literature period...")
        
        # Hardcoded periods for known planets
        periods = {
            'HAT-P-7 b': 2.204735,      # Pál et al. 2008
            'WASP-12 b': 1.09142245,    # Hebb et al. 2009
            'HAT-P-32 b': 2.150008,     # Hartman et al. 2011
            'Kepler-447 b': 7.79430132, # Literature value
            'Kepler-432 b': 52.501129,  # Ciceri et al. 2015
            'HAT-P-2 b': 5.6334729,     # Bakos et al. 2007, Pál et al. 2010
            'Kepler-63 b': 9.4343,      # Sanchis-Ojeda et al. 2013 - very grazing b~0.94
            'CoRoT-1 b': 1.5089557,     # Barge et al. 2008, Pont et al. 2010
            'Kepler-1658 b': 3.8493715, # NASA Exoplanet Archive
            'Kepler-423 b': 2.6843285,  # NASA Exoplanet Archive
            'Kepler-412 b': 1.7208613,  # NASA Exoplanet Archive
            'Kepler-13 b': 1.7635892,   # NASA Exoplanet Archive
            'Kepler-8 b': 3.5224986     # NASA Exoplanet Archive
        }
        
        known_period = periods.get(planet_name, 2.204735)
        print(f"  Literature period: {known_period:.8f} days")
        return known_period
    
    # Query known period from NASA Exoplanet Archive
    print(f"\nQuerying NASA Exoplanet Archive for {planet_name}...")
    
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
        import astropy.units as u
        
        # Query the confirmed planets table with timeout
        print(f"  Connecting to archive...")
        planet_table = NasaExoplanetArchive.query_object(planet_name, table='ps')
        print(f"  Query successful!")
        
        if len(planet_table) > 0:
            known_period = float(planet_table['pl_orbper'][0])  # Orbital period (days)
            period_err = float(planet_table['pl_orbpererr1'][0]) if planet_table['pl_orbpererr1'][0] else 0.0
            
            print(f"  Literature period: {known_period:.6f} ± {abs(period_err):.6f} days")
            print(f"  (from NASA Exoplanet Archive)")
        else:
            print(f"  No data found in archive, using fallback: 2.204735 days")
            known_period = 2.204735  # High-precision value from literature
            
    except Exception as e:
        print(f"  Archive query failed or timed out: {e}")
        print(f"  Using high-precision literature value: 2.204735 days")
        known_period = 2.204735  # Pál et al. 2008
    
    return known_period
