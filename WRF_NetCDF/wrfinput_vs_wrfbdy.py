field = inp['U'] # grid%u_2, dims=[bottom_top, south_north, west_east_stag]
msf = inp['MAPFAC_UY'] # grid%msfuy, dims=[south_north, west_east_stag]
mu = inp['MU'] # perturbation mu, grid%mu
mub = inp['MUB'] # base-state mu, grid%mub
c1 = inp['C1H'] # dims=[bottom_top]
c2 = inp['C2H'] # dims=[bottom_top]
field_bdy_xs = bdy['U_BXS'] # grid%u_bxs, dims=[bdy_width, bottom_top, south_north]
field_bdy_xe = bdy['U_BXE'] # grid%u_bxe, dims=[bdy_width, bottom_top, south_north]
field_bdy_ys = bdy['U_BYS'] # grid%u_bys, dims=[bdy_width, bottom_top, south_north]
field_bdy_ye = bdy['U_BYE'] # grid%u_bye, dims=[bdy_width, bottom_top, south_north]
field_bdy_tend_xs = bdy['U_BTXS'] # grid%u_btxs, dims=[bdy_width, bottom_top, south_north]
field_bdy_tend_xe = bdy['U_BTXE'] # grid%u_btxe, dims=[bdy_width, bottom_top, south_north]
field_bdy_tend_ys = bdy['U_BTYS'] # grid%u_btys, dims=[bdy_width, bottom_top, south_north]
field_bdy_tend_ye = bdy['U_BTYE'] # grid%u_btye, dims=[bdy_width, bottom_top, south_north]

# total mu at staggered u locs, set by rk_step_prep, which calls calc_mu_uv(mu,mub,...)
muu = np.zeros_like(msf)
muu[:,1:-1] = (mu[:,1:] + mu[:,:-1] + mub[:,1:] + mub[:,:-1]) / 2
muu[:,0] = mu[:,0] + mub[:,0]
muu[:,-1] = mu[:,-1] + mub[:,-1]
mut = xr.DataArray(muu, dims=msf.dims) # grid%muus == grid%muu at the beginning of the acoustic substepping, then updated with the instantaneous mu

#
# module_bc.spec_bdy_final()
#
dtbc = 0.0  # time delta since last boundary refresh
spec_zone = 5
bdy_dims = ('bottom_top','south_north','west_east_stag') # for u field

# field_bdy_ys
#   dims=(bdy_width: 5, bottom_top: 60, west_east_stag: 196)
bdy_width_dim = 'south_north'
bdy_transverse_dim = 'west_east_stag'
decoupled_field_bdy_ys = []
c1_2d = c1.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
c2_2d = c2.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
for j in range(spec_zone):
    bfield = field_bdy_ys.isel(bdy_width=j) \
           + dtbc * field_bdy_tend_ys.isel(bdy_width=j)
    xmsf = msf.isel({bdy_width_dim: j})
    xmu = c1_2d*mut.isel({bdy_width_dim: j}) + c2_2d
    bdy_plane = xmsf * bfield/xmu
    decoupled_field_bdy_ys.append(bdy_plane.expand_dims(bdy_width_dim))
decoupled_field_bdy_ys = xr.concat(decoupled_field_bdy_ys, bdy_width_dim)
decoupled_field_bdy_ys = decoupled_field_bdy_ys.transpose(*bdy_dims)
inp_field_bdy_ys = field.isel({bdy_width_dim: slice(0,spec_zone)})
assert np.all(decoupled_field_bdy_ys.XLAT_U == inp_field_bdy_ys.XLAT_U)
assert np.all(decoupled_field_bdy_ys.XLONG_U == inp_field_bdy_ys.XLONG_U)
assert np.allclose(decoupled_field_bdy_ys, inp_field_bdy_ys)

# field_bdy_ye
#   dims=(bdy_width: 5, bottom_top: 60, west_east_stag: 196)
bdy_width_dim = 'south_north'
bdy_transverse_dim = 'west_east_stag'
decoupled_field_bdy_ye = []
c1_2d = c1.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
c2_2d = c2.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
for j in range(spec_zone-1,-1,-1):
    bfield = field_bdy_ye.isel(bdy_width=j) \
           + dtbc * field_bdy_tend_ye.isel(bdy_width=j)
    xmsf = msf.isel({bdy_width_dim: -j-1})
    xmu = c1_2d*mut.isel({bdy_width_dim: -j-1}) + c2_2d
    bdy_plane = xmsf * bfield/xmu
    decoupled_field_bdy_ye.append(bdy_plane.expand_dims(bdy_width_dim))
decoupled_field_bdy_ye = xr.concat(decoupled_field_bdy_ye, bdy_width_dim)
decoupled_field_bdy_ye = decoupled_field_bdy_ye.transpose(*bdy_dims)
inp_field_bdy_ye = field.isel({bdy_width_dim: slice(-spec_zone,None)})
assert np.all(decoupled_field_bdy_ye.XLAT_U == inp_field_bdy_ye.XLAT_U)
assert np.all(decoupled_field_bdy_ye.XLONG_U == inp_field_bdy_ye.XLONG_U)
assert np.allclose(decoupled_field_bdy_ye, inp_field_bdy_ye)

# field_bdy_xs
#   dims=(bdy_width: 5, bottom_top: 60, south_north: 196)
bdy_width_dim = 'west_east_stag'
bdy_transverse_dim = 'south_north'
decoupled_field_bdy_xs = []
c1_2d = c1.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
c2_2d = c2.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
for i in range(spec_zone):
    bfield = field_bdy_xs.isel(bdy_width=i) \
           + dtbc * field_bdy_tend_xs.isel(bdy_width=i)
    xmsf = msf.isel({bdy_width_dim: i})
    xmu = c1_2d*mut.isel({bdy_width_dim: i}) + c2_2d
    bdy_plane = xmsf * bfield/xmu
    decoupled_field_bdy_xs.append(bdy_plane.expand_dims(bdy_width_dim))
decoupled_field_bdy_xs = xr.concat(decoupled_field_bdy_xs, bdy_width_dim)
decoupled_field_bdy_xs = decoupled_field_bdy_xs.transpose(*bdy_dims)
inp_field_bdy_xs = field.isel({bdy_width_dim: slice(0,spec_zone)})
assert np.all(decoupled_field_bdy_xs.XLAT_U == inp_field_bdy_xs.XLAT_U)
assert np.all(decoupled_field_bdy_xs.XLONG_U == inp_field_bdy_xs.XLONG_U)
assert np.allclose(decoupled_field_bdy_xs, inp_field_bdy_xs)

# field_bdy_xe
#   dims=(bdy_width: 5, bottom_top: 60, south_north: 196)
bdy_width_dim = 'west_east_stag'
bdy_transverse_dim = 'south_north'
decoupled_field_bdy_xe = []
c1_2d = c1.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
c2_2d = c2.expand_dims({bdy_transverse_dim: msf.sizes[bdy_transverse_dim]})
for i in range(spec_zone-1,-1,-1):
    bfield = field_bdy_xe.isel(bdy_width=i) \
           + dtbc * field_bdy_tend_xe.isel(bdy_width=i)
    xmsf = msf.isel({bdy_width_dim: -i-1})
    xmu = c1_2d*mut.isel({bdy_width_dim: -i-1}) + c2_2d
    bdy_plane = xmsf * bfield/xmu
    decoupled_field_bdy_xe.append(bdy_plane.expand_dims(bdy_width_dim))
decoupled_field_bdy_xe = xr.concat(decoupled_field_bdy_xe, bdy_width_dim)
decoupled_field_bdy_xe = decoupled_field_bdy_xe.transpose(*bdy_dims)
inp_field_bdy_xe = field.isel({bdy_width_dim: slice(-spec_zone,None)})
assert np.all(decoupled_field_bdy_xe.XLAT_U == inp_field_bdy_xe.XLAT_U)
assert np.all(decoupled_field_bdy_xe.XLONG_U == inp_field_bdy_xe.XLONG_U)
assert np.allclose(decoupled_field_bdy_xe, inp_field_bdy_xe)