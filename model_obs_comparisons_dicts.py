import numpy as np
import xarray as xr
# import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import glob
import xesmf as xe
import matplotlib.colors as colors


kg_per_m3 = 1026.0


nitrate_dict = dict(
    var = 'NO3',
    var_name = 'nitrate',
    model_var_name = 'NO3',
    units = r'$mmol\ m^{-3}$',
    model_conv = 1,
    WOA_dir = '/global/cfs/cdirs/m4632/shared_datasets/WOA23/nitrate/',
    WOA_file_str = 'woa23_all_n', 
    WOA_var_name = 'n_an',
    WOA_conv = 1,
    GLODAP_dir = '/global/cfs/cdirs/m4632/shared_datasets/GLODAP/GLODAPv2.2016b_MappedClimatologies/',
    GLODAP_file_name = 'GLODAPv2.2016b.NO3.nc',
    GLODAP_var_name = 'NO3',
    GLODAP_conv = 1,
    plot_kwargs = dict(
                    # vmin = 0, 
        #           vmax = 10,
                  # levels = np.linspace(0,10,21),
                  cmap = 'Spectral_r',
                  extend = 'both',
              )
)

phosphate_dict = dict(
    var = 'PO4',
    var_name = 'phosphate',
    model_var_name = 'PO4',
    units = r'$mmol\ m^{-3}$',
    model_conv = 1,
    WOA_dir = '/global/cfs/cdirs/m4632/shared_datasets/WOA23/phosphate/',
    WOA_file_str = 'woa23_all_p',  
    WOA_var_name = 'p_an',
    WOA_conv = 1,
    GLODAP_dir = '/global/cfs/cdirs/m4632/shared_datasets/GLODAP/GLODAPv2.2016b_MappedClimatologies/',
    GLODAP_file_name = 'GLODAPv2.2016b.PO4.nc',
    GLODAP_var_name = 'PO4',
    GLODAP_conv = 1,
    plot_kwargs = dict(
                  # vmin = 0, 
                  # vmax = 1,
                  # levels = np.linspace(0,1,21),
                  cmap = 'Spectral_r',
                  extend = 'both',
              )
)

silicate_dict = dict(
    var = 'SiO3',
    var_name = 'silicate',
    model_var_name = 'SiO3',
    units = r'$mmol\ m^{-3}$',
    model_conv = 1,
    WOA_dir = '/global/cfs/cdirs/m4632/shared_datasets/WOA23/silicate/',
    WOA_file_str = 'woa23_all_i',    
    WOA_var_name = 'i_an',
    WOA_conv = 1,
    GLODAP_dir = '/global/cfs/cdirs/m4632/shared_datasets/GLODAP/GLODAPv2.2016b_MappedClimatologies/',
    GLODAP_file_name = 'GLODAPv2.2016b.silicate.nc',
    GLODAP_var_name = 'silicate',
    GLODAP_conv = 1,
    plot_kwargs = dict(
                  # vmin = 0, 
                  # vmax = 24,
                  # levels = np.linspace(0,24,25),
                  cmap = 'Spectral_r',
                  extend = 'both',
              )
)

oxygen_dict = dict(
    var = 'O2',
    var_name = 'oxygen',
    model_var_name = 'O2',
    units = r'$mmol\ m^{-3}$',
    model_conv = 1,
    WOA_dir = '/global/cfs/cdirs/m4632/shared_datasets/WOA23/oxygen/',
    WOA_file_str = 'woa23_all_o',    
    WOA_var_name = 'o_an',
    WOA_conv = 1,
    GLODAP_dir = '/global/cfs/cdirs/m4632/shared_datasets/GLODAP/GLODAPv2.2016b_MappedClimatologies/',
    GLODAP_file_name = 'GLODAPv2.2016b.oxygen.nc',
    GLODAP_var_name = 'oxygen',
    GLODAP_conv = 1,
    plot_kwargs = dict(
                  # vmin = 200, 
                  # vmax = 350,
                  # levels = np.linspace(200,350,16),
                  cmap = 'Spectral_r',
                  extend = 'both',
              )
)

alkalinity_dict = dict(
    var = 'Alk',
    var_name = 'alkalinity',
    model_var_name = 'ALK',
    units = r'$\mu mol\ kg^{-1}$',
    model_conv = kg_per_m3/1000,
    GLODAP_dir = '/global/cfs/cdirs/m4632/shared_datasets/GLODAP/GLODAPv2.2016b_MappedClimatologies/',
    GLODAP_file_name = 'GLODAPv2.2016b.TAlk.nc',
    GLODAP_var_name = 'TAlk',
    GLODAP_conv = 1,
    plot_kwargs = dict(
                  # vmin = 2150, 
                  # vmax = 2550,
                  # levels = np.linspace(2150,2450,21),
                  cmap = 'Spectral_r',
                  extend = 'both',
              )
)

DIC_dict = dict(
    var = 'DIC',
    var_name = 'total DIC',
    model_var_name = 'DIC',
    units = r'$mmol\ m^{-3}$',
    model_conv = kg_per_m3/1000,
    GLODAP_dir = '/global/cfs/cdirs/m4632/shared_datasets/GLODAP/GLODAPv2.2016b_MappedClimatologies/',
    GLODAP_file_name = 'GLODAPv2.2016b.TCO2.nc',
    GLODAP_var_name = 'TCO2',
    GLODAP_conv = 1,
    plot_kwargs = dict(
                  # vmin = 1900, 
                  # vmax = 2200,
                  # levels = np.linspace(1900,2200,31),
                  cmap = 'Spectral_r',
                  extend = 'both',
              )
)

Chl_dict = dict(
    var = 'Chl',
    var_name = 'chlorophyll',
    model_var_list = ['spChl','diatChl', 'diazChl'],
    units = r'$mg\ m^{-3}$',
    model_conv = 1,
    MODIS_dir = '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/',
    MODSIS_file_list = ['/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20030101_20240131.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20030201_20240229.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20030301_20240331.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20030401_20230430.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20030501_20230531.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20030601_20230630.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20020701_20230731.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20020801_20230831.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20020901_20230930.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20021001_20231031.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20021101_20231130.L3m.MC.CHL.chlor_a.4km.nc',
             '/global/cfs/cdirs/m4632/shared_datasets/aqua_modis/AQUA_MODIS.20021201_20231231.L3m.MC.CHL.chlor_a.4km.nc',
                       ],
    MODIS_var_name = 'chlor_a',
    MODIS_conv = 1,
    plot_kwargs = dict(vmin = .01, 
                  vmax = 10,
                  cmap = 'Spectral_r',
                  extend = 'both',
                  levels = np.logspace(np.log10(.01),np.log10(10), 19),
                  norm=colors.LogNorm(),
              )
)


