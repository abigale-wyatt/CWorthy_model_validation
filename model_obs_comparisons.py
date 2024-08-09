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
from math import log10, floor
from xgcm import Grid
import warnings



def myround(x, base=5):
    return 5 * round(x/5)


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))
def round_to_2(x):
    return round(x, -int(np.floor(np.log10(abs(x)))) +1)


def set_model_time(model_DS):
    
    ds = model_DS.ocean_time.isel(time=0)
    times = pd.to_datetime(ds, unit='s', origin = pd.Timestamp("1997-01-01 00:00:00")) 
    model_DS['time_counter'] = times
    
    return model_DS
    

def get_relevant_timeframe(model_DS, obs_DS, verbose = True):
    
    if verbose == True:
        print('...calculating comparison time frame...')
    
    ds = set_model_time(model_DS)
    
    with xr.set_options(keep_attrs=True):
        model_clim = ds.groupby('time_counter.month').mean('time_counter')
    mths_of_interest = model_clim.month

    if 'time' in obs_DS.coords:
        mths = np.ceil(obs_DS.time % 12)
        obs_DS['time'] = mths 
        obs_DS = obs_DS.rename({'time':'month'})
        
    if 'month' in obs_DS.coords:
        obs_DS = obs_DS.sel(month = slice(mths_of_interest[0].item(), mths_of_interest[-1].item()))
    
    return model_clim, obs_DS


def land_map_plot(ax,
                     xmin=-180,
                     xmax=180,
                     ymin=0,
                     ymax=65,
                     title='Title',
                     projection = ccrs.PlateCarree(),
                     verbose=True,
                    ):
    
    if verbose==True:
        print('...adding pretty features...')
    
    xstep = myround( (xmax-xmin)/3)
    ystep = myround( (ymax-ymin)/4)
    
    ax.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')
    ax.set_xticks(np.arange(myround(xmin), myround(xmax), xstep), crs=projection)
    ax.set_yticks(np.arange(myround(ymin), myround(ymax), ystep), crs=projection)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_facecolor('.95')
    ax.set_title(title)
    
    
def plot_box(ax, domain_corners_list,
            projection = ccrs.PlateCarree()
            ):

    p1, p2, p3, p4 = domain_corners_list

    left   = ax.plot([p1[0], p2[0]],[p1[1], p2[1]],
                     transform=projection, color='k', marker='.', ls='--')
    right  = ax.plot([p2[0], p3[0]],[p2[1], p3[1]],
                     transform=projection, color='k', marker='.', ls='--')
    top    = ax.plot([p3[0], p4[0]],[p3[1], p4[1]],
                     transform=projection, color='k', marker='.', ls='--')
    bottom = ax.plot([p4[0], p1[0]],[p4[1], p1[1]],
                     transform=projection, color='k', marker='.', ls='--')

    
def get_model_bounds(model_grd):
    p1 = model_grd.isel(xi_rho= 0,  eta_rho= 0)
    p1 = (p1.lon_rho.values.item(), p1.lat_rho.values.item())
    
    p2 = model_grd.isel(xi_rho= 0,  eta_rho= -1)
    p2 = (p2.lon_rho.values.item(), p2.lat_rho.values.item())
    
    p3 = model_grd.isel(xi_rho= -1, eta_rho= -1)
    p3 = (p3.lon_rho.values.item(), p3.lat_rho.values.item())
    
    p4 = model_grd.isel(xi_rho= -1, eta_rho= 0)
    p4 = (p4.lon_rho.values.item(), p4.lat_rho.values.item())
    
    domain_corners_list = [p1,p2,p3,p4]
    return domain_corners_list


def get_model_area(model_grd):
        dx = 1/model_grd.pm; dy = 1/model_grd.pn;
        model_area=dx*dy
        return model_area
    
    
def adjust_x_lon(xvals):
    xvals = xr.where(xvals > 180, (xvals % -180), xvals)
    return xvals


def get_z_vals(model_DS, model_grd, verbose=True, Vtransform=4):
    
    if verbose==True:
        print('...stretching ROMS vertical coordinate...')

    ds = model_DS
    ds.coords["s_rho"] = np.linspace(-1,0,100)
    ds = ds.sortby('s_rho')

    Cs = xr.DataArray(ds.Cs_r, dims='s_rho')

    if Vtransform == 1:
        Zo_rho = ds.hc * (ds.s_rho - Cs) + Cs * model_grd.h
        z_rho = Zo_rho + ds.zeta * (1 + Zo_rho / model_grd.h)

    if Vtransform == 2:
        Zo_rho = (ds.hc * ds.s_rho + Cs * model_grd.h) / (ds.hc + model_grd.h)
        z_rho = ds.zeta + (ds.zeta + model_grd.h) * Zo_rho
        
    if Vtransform == 4: 
        theta_s = ds.theta_s
        theta_b = ds.theta_b
        
        C = (1 - np.cosh(theta_s * ds.s_rho)) / (np.cosh(theta_s) - 1)
        Cs = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        
        Zo_rho = (ds.hc * ds.s_rho + Cs * model_grd.h) / (ds.hc + model_grd.h)
        z_rho = ds.zeta + (ds.zeta + model_grd.h) * Zo_rho
    
    ds.coords["z_rho"] = z_rho.transpose()
    
    return ds



def get_dz(z_vals_model): 
    dz = z_vals_model.z_rho.diff('s_rho')
    dz.name = 'dz'
    return dz


def model_zgridded(model_DS, model_grd, model_var_name, 
                   target_zs = np.concatenate ( (np.arange(0,-200,-1), np.arange(-200,-5000,-50)) ), 
                   verbose=True):
    
    z_vals_model = get_z_vals(model_DS, model_grd, verbose=verbose)
    var_ds = z_vals_model[model_var_name].load()
    
    if verbose == True:
        print('...regridding ROMS vertical coordinates...')
    
    grid = Grid(z_vals_model, coords={'s_rho': {'center':'s_rho'}}, periodic=False)
    
    with warnings.catch_warnings(action="ignore"):
        transformed = grid.transform(var_ds, 's_rho', target_zs, target_data=z_vals_model.z_rho)
    
    del var_ds
    
    return transformed


def side_by_side_plot(model_grd, 
                      model_DS,
                      obs_DS,
                      units = 'test',
                      obs_x = 'lon',
                      obs_y = 'lat',
                      model_x = 'lon_rho',
                      model_y = 'lat_rho',
                      figsize=(8,8),
                      projection = ccrs.PlateCarree(),
                      title1='Title1',
                      title2='Title2',
                      verbose = True, 
                      **var_kwargs,           
):
    if verbose==True:
        print('starting figure...')
    
    fig,axes = plt.subplots(1,3, figsize=figsize, subplot_kw={'projection': projection})
    
    if 'vmax' not in var_kwargs.keys():
        if verbose==True:
            print('...calculating vmax...')
            
        vmax = round_to_2(
                np.nanquantile(obs_DS.values,.9 ))
        vmin = round_to_2(
                np.nanquantile(obs_DS.values, .1 ))
        lvls = np.linspace(vmin,vmax, 15)
        var_kwargs.update(dict(vmin=vmin, vmax=vmax, levels=lvls))

    if verbose==True:
        print('...plotting observed field...')
        
    cont_model = axes[0].contourf(model_grd[model_x], model_grd[model_y], model_DS,
                                  **var_kwargs)
    cont_obs = axes[1].contourf(obs_DS[obs_x], obs_DS[obs_y], obs_DS, 
                                  **var_kwargs)
    
    if verbose==True:
        print('...plotting modeled field...') 
        
    model_regridded = regrid_model_to_obs(model_DS, model_grd, obs_DS, verbose=verbose)
    diff = (model_regridded - obs_DS.where(model_regridded !=0))
    
    # Get boundaries and levels for the difference colormap
    diff_max = abs(round_to_1( diff.max().values * 0.8 ))

    center = 0
    if diff_max % 2 == 0:
        center = 1
    lvls = np.linspace(-diff_max, diff_max, 20 + center)
        
    cont_diff = axes[2].contourf(diff.lon, diff.lat, diff, 
                     vmin = -diff_max, 
                     vmax = diff_max, 
                     levels = lvls,
                     cmap = 'RdBu_r',
                     extend = 'both')

    cax1 = fig.add_axes([0.05, 0.38, 0.02, 0.22], label=units)
    cbar1 = fig.colorbar(cont_obs, cax=cax1, label=units)
    cax1.yaxis.set_ticks_position('left')
    cax1.yaxis.set_label_position('left')
    
    cax2 = fig.add_axes([0.92, 0.38, 0.02, 0.22], label=units)
    cbar2 = fig.colorbar(cont_diff, cax=cax2, label=units)

    model_corners = get_model_bounds(model_grd)
    
    xmin = adjust_x_lon( min([i[0] for i in model_corners]) )
    xmax = adjust_x_lon( max([i[0] for i in model_corners]) )
    ymin = min([i[1] for i in model_corners])
    ymax = max([i[1] for i in model_corners])
    
    x_buffer = (xmax-xmin)*.02
    y_buffer = (ymax-ymin)*.02

    for ax, title in zip(axes, [title1, title2, '(Model - Obs)']):
        land_map_plot(ax, 
                         xmin = xmin - x_buffer,
                         xmax = xmax + x_buffer, 
                         ymin = ymin - y_buffer, 
                         ymax = ymax + y_buffer,
                         title = title, 
                         verbose=verbose,
                    )  
        plot_box(ax,model_corners) 
    return fig


def regrid_model_to_obs (modelDS, model_grd, obsDS, verbose=True):
    
    if verbose==True:
        print('...matching horiztonal grids...')
    
    grd = model_grd.rename({'lat_rho':'lat', 'lon_rho':'lon'})[['lat','lon']]
    regridder = xe.Regridder(grd, obsDS, "bilinear")
    
    return (regridder(modelDS, output_chunks=dict(lat=len(obsDS.lat), lon=len(obsDS.lon))))

    
def WOA_2D_side_by_side(var_dict, model_DS, model_grd, d=1, verbose=False):
    var_name = var_dict['var_name']
    
    WOA_dir = var_dict['WOA_dir']
    file_str = var_dict['WOA_file_str']
    obs_file_list = [WOA_dir + file_str + str(s).zfill(2) + '_01.nc' for s in range(1,13) ]
    WOA_var_name = var_dict['WOA_var_name']

    model_var_name = var_dict['model_var_name']
    var_plot_kwargs = var_dict['plot_kwargs']
    units = var_dict['units']
    
    if verbose==True:
        print('standardizing and regridding models and obs datasets...')
    
    obs_DS = xr.open_mfdataset(obs_file_list, decode_times=False) * var_dict['WOA_conv']
    model_clim, obs_DS = get_relevant_timeframe(model_DS, obs_DS, verbose = verbose)
    obs = obs_DS[WOA_var_name].mean('month').interp(depth=d)
    obs = obs.where(obs != 0, np.nan)
    
    model_clim_var = model_clim.where(model_clim != 0, np.nan)[[model_var_name, 'zeta']].isel(time=1)  
    model_clim_var = model_clim_var.assign_attrs(model_clim.attrs) #return attributes from larger dataset
    z_transformd_model = model_zgridded(model_clim_var, model_grd, model_var_name, verbose=verbose)
    model = z_transformd_model.sel(z_rho = -d, method='nearest').mean('month')
    
    dstr = str(d) + 'm '
    if d < 10:
        dstr = 'sfc '
        
    fig = side_by_side_plot(model_grd = model_grd,
                          model_DS = model,
                          obs_DS = obs,
                          title1 = 'Model ' + dstr + var_name,
                          title2 = 'WOA ' + dstr + var_name,
                          units = units,
                          verbose = verbose, 
                          **var_plot_kwargs,
                     )
    return fig


def GLODAP_2D_side_by_side(var_dict, model_DS, model_grd, d=1, verbose=False):
    var_name = var_dict['var_name']
    
    GLODAP_dir = var_dict['GLODAP_dir']
    file_name = var_dict['GLODAP_file_name']
    GLODAP_var_name = var_dict['GLODAP_var_name']

    model_var_name = var_dict['model_var_name']
    var_plot_kwargs = var_dict['plot_kwargs']
    units = var_dict['units']

    if verbose==True:
        print('standardizing models and obs datasets...')

    obs_DS = xr.open_mfdataset(GLODAP_dir + file_name, decode_times=False) * var_dict['GLODAP_conv']
    obs_DS['depth_surface'] = obs_DS.Depth
    obs = obs_DS[GLODAP_var_name].interp(depth_surface=d)
    model_clim, _ = get_relevant_timeframe(model_DS, model_DS, verbose = verbose)


    model_clim_var = model_clim.where(model_clim != 0, np.nan)[[model_var_name, 'zeta']].isel(time=1)  
    model_clim_var = model_clim_var.assign_attrs(model_clim.attrs) #return attributes from larger dataset
    z_transformd_model = model_zgridded(model_clim_var, model_grd, model_var_name, verbose=verbose)
    model = z_transformd_model.sel(z_rho = -d, method='nearest').mean('month') * var_dict['model_conv']
    
    dstr = str(d) + 'm '
    if d < 10:
        dstr = 'sfc '
    
    fig = side_by_side_plot(model_grd = model_grd,
                          model_DS = model,
                          obs_DS = obs,
                          title1 = 'Model ' + dstr + var_name,
                          title2 = 'GLODAP ' + dstr + var_name,
                          units = units,
                          verbose = verbose, 
                          **var_plot_kwargs,
                     )
    return fig


def AquaMODIS_side_by_side(var_dict, model_DS, model_grd, verbose=False):
    var_name = var_dict['var_name']

    MODIS_dir = var_dict['MODIS_dir']
    file_list = var_dict['MODSIS_file_list']
    MODIS_var_name = var_dict['MODIS_var_name']

    var_list = var_dict['model_var_list']
    var_name = var_dict['var_name']
    var_plot_kwargs = var_dict['plot_kwargs']
    units = var_dict['units']
        
    if verbose==True:
        print('standardizing and regridding models and obs datasets...')

    obs_DS = xr.open_mfdataset(file_list, combine='nested', concat_dim='month') * var_dict['MODIS_conv']
    obs_DS['month'] = range(1,13,1)
    model_clim, obs_DS = get_relevant_timeframe(model_DS, obs_DS, verbose = verbose)

    obs = obs_DS[MODIS_var_name].mean('month')
    obs = obs.where(obs>0) # make sure there are no negative values for the log scale

    # Take first model layer as surface
    model_clim = model_clim.where(model_clim != 0, np.nan)
    model = model_clim[var_list].to_array().sum("variable").isel(time=1,s_rho=-1).mean('month') * var_dict['model_conv']

    fig = side_by_side_plot(model_grd = model_grd,
                          model_DS = model,
                          obs_DS = obs,
                          title1 = 'Model sfc ' + var_name,
                          title2 = 'AquaMODIS sfc ' + var_name,
                          units = units,
                          verbose = verbose, 
                          **var_plot_kwargs,
                     )

    return fig
