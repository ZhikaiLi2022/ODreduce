import os
import glob
import copy
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import astropy.io.fits as fits
import astropy.stats as stats
import photutils as pu
import matplotlib.pyplot as plt


from odreduce import __file__
from odreduce.plots import set_plot_params
from odreduce.models import *


#####################################################################
# FUNCTIONALITY OF THE SOFTWARE
#

def get_info(args):
    # Get parameters for all modules
    args = get_main_params(args)
    set_plot_params()

    return args

def get_main_params(args, stars=None, verbose=False, command='run', show=False, 
                    save=True):
    vars = ['stars', 'inpdir', 'outdir', 'cli', 'command', 'info', 'show', 'save',
            'verbose', 'wave_band']
    if args.cli:
        vals = [args.stars, args.inpdir, args.outdir, args.cli, args.command, args.info,
                args.show, args.save,args.verbose, args.wave_band]
    args.params = dict(zip(vars,vals))
    return args


#####################################################################
# Data and information related to a processed star
#


def load_data(star, args):
    """
    """
    if glob.glob(os.path.join(args.inpdir, 'bias*')) != []:
        if glob.glob(os.path.join(args.inpdir, 'flat*')) != []:
            if glob.glob(os.path.join(args.inpdir, '%s*'%str(star.name))) != []:
                if star.verbose:
                    print('\n\n------------------------------------------------------')
                    print('Target: %s'%str(star.name[0]))
                    print('------------------------------------------------------')
                # Load bias and flat, maybe have data of TPF
                args, star, note = load_bias(args, star)
                if star.verbose:
                    print(note)
                    print('------------------------------------------------------')
                args, star, note = load_flat(args, star)

                if star.verbose:
                    print(note)
                    print('------------------------------------------------------\n')
                # create results directory
                if not os.path.exists(args.outdir):
                    os.mkdir(args.outdir)
                args, star, note = load_ffi(args, star)
                if star.verbose:
                    print(note)
                    print('------------------------------------------------------')
                args, star, note = cross_star(args, star)
                if star.verbose:
                    print(note)
                    print('------------------------------------------------------')
    return star

def load_bias(args, star, note=''):
    star.bias = True
    # Try loading the bias
    bias_lst = glob.glob(os.path.join(args.inpdir, 'bias*'))
    star.bias = True
    n_bias = len(bias_lst)
    nx = fits.getval(bias_lst[0], 'NAXIS1')
    ny = fits.getval(bias_lst[0], 'NAXIS2')

    bias_cube = np.empty((n_bias, ny,nx), dtype='uint16')
    for i in range(n_bias):
        bias_cube[i] = fits.getdata(bias_lst[i])
    star.mbias = np.float32(np.median(bias_cube, axis=0))
    note += '#BIAS: %d X %d array of data read\n#'%(nx,ny)
    star.nx = nx
    star.ny = ny
    return args, star, note

def load_flat(args, star, note=''):
    if args.wave_band:
        flat_lst = glob.glob(os.path.join(args.inpdir, 'flat_%s*'%args.wave_band))
        n_flat = len(flat_lst)
        nx = fits.getval(flat_lst[0], 'NAXIS1')
        ny = fits.getval(flat_lst[0], 'NAXIS2')
        flat_cube = np.empty((n_flat, ny, nx), dtype='float32')
        for i in range(n_flat):
            flat_tmp = fits.getdata(flat_lst[i])
            flat_tmp = flat_tmp - star.mbias
            flat_med = np.median(flat_tmp)
            flat_cube[i] = flat_tmp / flat_med
        star.mflat = np.median(flat_cube, axis=0)
        note += '#FLAT:wave band %s ,%d X %d array of data read\n#'%(args.wave_band,nx,ny)
    return args, star, note

def load_ffi(args, star, note=''):
    obj_lst = glob.glob(os.path.join(args.inpdir, '%s_%s*'%(star.name[0], args.wave_band)))
    n_obj= len(obj_lst)
    for i in range(n_obj):

        if i == 0:
            from tqdm import tqdm
            pbar = tqdm(total=n_obj)
        pbar.update(1)
        if i == n_obj-1:
            pbar.close()

        if not os.path.exists(os.path.join(args.outdir, f"{obj_lst[i][len(args.inpdir)+1:-4]}.bf.fits")) or not os.path.exists(os.path.join(args.outdir, f"{obj_lst[i][len(args.inpdir)+1:-4]}.cat.fits")):
            dathdl = fits.open(obj_lst[i])
            rawimg = dathdl[0].data
            rawhdr = dathdl[0].header
            corimg = (rawimg - star.mbias) / star.mflat
            raw_cli = stats.sigma_clip(rawimg, masked=False)
            raw_med = np.median(raw_cli)
            raw_std = np.std(raw_cli)
            cor_cli = stats.sigma_clip(corimg, masked=False)
            cor_med = np.median(cor_cli)
            cor_std = np.std(cor_cli)
            # save corimg to fits
            corhdr = rawhdr.copy()
            ss = f"{corhdr}"
            cor_hl = fits.HDUList([fits.PrimaryHDU(data=corimg, header=corhdr)])
            cor_hl.writeto(os.path.join(args.outdir, f"{obj_lst[i][len(args.inpdir)+1:-4]}.bf.fits"), overwrite=True)
            dathdl.close()
            if args.find_target:
                iraffind = pu.detection.IRAFStarFinder(fwhm=3.0, threshold=10.*cor_std)
                bkg = co_bkg(corimg)
                sourcesi = iraffind(corimg - bkg.background_median)
                fwhm = np.mean(stats.sigma_clip(sourcesi["fwhm"], masked=False))
                
                pos = np.transpose((sourcesi['xcentroid'], sourcesi['ycentroid']))
                aper1 = pu.aperture.CircularAperture(pos, r=2.0 * fwhm)
                phot1 = pu.aperture.aperture_photometry(corimg - bkg.background, aper1, error=bkg.background_rms)
                flux = phot1["aperture_sum"]
                fluxerr = phot1["aperture_sum_err"]
                mag = 25 - 2.5 * np.log10(flux)
                magerr = 2.5 * np.log10((flux + fluxerr) / flux)

                catdt = [
                    ("x",    np.float64),
                    ("y",    np.float64),
                    ("fwhm", np.float32),
                    ("flux", np.float64),
                    ("ferr", np.float64),
                    ("mag",  np.float32),
                    ("merr", np.float32),
                ]
                cat = np.empty(len(sourcesi), dtype=catdt)
                cat["x"   ] = sourcesi['xcentroid']
                cat["y"   ] = sourcesi['ycentroid']
                cat["fwhm"] = sourcesi['fwhm']
                cat["flux"] = flux
                cat["ferr"] = fluxerr
                cat["mag" ] = mag
                cat["merr"] = magerr
                cat_hl = fits.HDUList([
                    fits.PrimaryHDU(header=corhdr),
                    fits.BinTableHDU(data=cat),
                ])
                cat_hl.writeto(os.path.join(args.outdir, f"{obj_lst[i][len(args.inpdir)+1:-4]}.cat.fits"), overwrite=True)
         
    note += '\n# %s FFI:wave band %s , pixel file is saved to outdir#'%(star.name, args.wave_band)
    return args, star, note

def co_bkg(corimg):
    sigma_clip = stats.SigmaClip(sigma=3.)
    bkg_estimator = pu.background.MedianBackground()
    bkg = pu.background.Background2D(corimg, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return bkg

def cross_star(args, star, note=''):
    sigmaclipmean = lambda dat: np.mean(stats.sigma_clip(dat, masked=False))
    sigmaclipstd = lambda dat: np.std(stats.sigma_clip(dat, masked=False))
    cat_lst = glob.glob(os.path.join(args.outdir, '%s_%s*cat*'%(star.name[0], args.wave_band)))
    cat_lst = sorted(cat_lst, key=str.lower)
    n_cat = len(cat_lst)
    '''
    n_cross = np,random.randint(0, n_cat, size=(2,2))
    '''
    nni = 0
    cati = fits.getdata(cat_lst[0])
    xi = cati["x"]
    yi = cati["y"]
    ni = len(cati)
    matmag = np.zeros([n_cat, ni], np.float32) + np.nan
    for j in range(1,n_cat):
        catj = fits.getdata(cat_lst[j])
        xj = catj["x"]
        yj = catj["y"]
        nj = len(catj)
        xij = np.array([xi] * nj) - np.array([xj] * ni).T
        yij = np.array([yi] * nj) - np.array([yj] * ni).T
        xhist = np.histogram(xij.flatten(), bins=200, range=(-100, 100))
        yhist = np.histogram(yij.flatten(), bins=200, range=(-100, 100))
        x_peak_ix = np.argmax(xhist[0])
        y_peak_ix = np.argmax(yhist[0])
        x_diff_l, x_diff_h = xhist[1][x_peak_ix], xhist[1][x_peak_ix+1]
        y_diff_l, y_diff_h = yhist[1][y_peak_ix], yhist[1][y_peak_ix+1]
        x_hist2 = np.histogram(xij.flatten(), bins=100, range=(2*x_diff_l-x_diff_h, 2*x_diff_h-x_diff_l))
        y_hist2 = np.histogram(yij.flatten(), bins=100, range=(2*y_diff_l-y_diff_h, 2*y_diff_h-y_diff_l))
        x_peak_ix = np.argmax(x_hist2[0])
        y_peak_iy = np.argmax(y_hist2[0])
        x_diff = (x_hist2[1][x_peak_ix] + x_hist2[1][x_peak_ix+1]) / 2
        y_diff = (y_hist2[1][y_peak_iy] + y_hist2[1][y_peak_iy+1]) / 2
        xij_clip = xij[np.abs(xij - x_diff) < 6].flatten()
        yij_clip = yij[np.abs(yij - y_diff) < 6].flatten()
        x_diff, x_std = sigmaclipmeanstd(xij_clip)
        y_diff, y_std = sigmaclipmeanstd(yij_clip)
        xyij2 = (xij - x_diff) ** 2 + (yij - y_diff) ** 2
        ijix = np.where(xyij2 < 5 * 5)
        jix, iix = ijix
        if j == 1 :
            matmag[nni] = cati["mag"]
        matmag[j, iix] = catj["mag"][jix]
        mag_diff = cati["mag"][iix]  - catj["mag"][jix]
        mag_d, mag_s = sigmaclipmeanstd(mag_diff)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(xi, yi, "r+")
    ax.plot(xj, yj, "gx")
    for v, u in zip(ijix[0], ijix[1]):
        ax.plot([xi[u], xj[v]], [yi[u], yj[v]], "r")
    plt.show()
    matmag = matmag.T
    mag_diff = matmag - np.array([matmag[:, 0]]*n_cat).T
    mag_d = np.nanmedian(mag_diff, axis=0)
    plt.plot(mag_d)
    plt.show()
    matmag_cali = matmag - np.array([mag_d] * ni)
    matmag_cali = [i for i in matmag_cali if c_nan(i)<0.05]
    star_ref1, small = find_refstars(matmag_cali)
    fig, ax = plt.subplots(figsize=(10, 30))
    for i in range(len(small)):
        ax.plot(matmag_cali[small[i][0]] - np.nanmedian(matmag_cali[small[i][0]]) + i)
    ax.set_ylim(len(small), -1)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 30))
    for i in range(len(star_ref1)):
        ax.plot(matmag_cali[star_ref1[i]] - np.nanmedian(matmag_cali[star_ref1[i]]) + i)
    ax.set_ylim(len(star_ref1), -1)
    plt.show()

    matmag_cali_ref = np.array([matmag_cali[i] for i in star_ref1]).T
    matmag_cali_ref = np.array([np.mean(i) for i in matmag_cali_ref]).T
    for i in range(len(star_ref1)):
        ax.plot(matmag_cali[star_ref1[i]] - np.nanmedian(matmag_cali[star_ref1[i]]) + i)
    ax.set_ylim(len(star_ref1), -1)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 30))
    if args.wave_band == ['V']:
        plt.scatter([i for i in range(len(matmag_cali_ref))], [matmag_cali[small[1][0]][i] - matmag_cali_ref[i] for i in range(len(matmag_cali_ref))], label='Target')
    else:
        plt.scatter([i for i in range(len(matmag_cali_ref))], [matmag_cali[small[0][0]][i] - matmag_cali_ref[i] for i in range(len(matmag_cali_ref))], label='Target')
    for j in range(len(star_ref1)):
        check_star = []
        for i in range(len(matmag_cali_ref)):
            #check_v = matmag_cali[star_ref1[1]][i] - matmag_cali_ref[i]
            check_v = matmag_cali[star_ref1[j]][i] - matmag_cali_ref[i]
            check_star.append(check_v)
        ax.scatter([i for i in range(len(check_star))], check_star,label=r'$\sigma =%.7s$'%str(np.nanstd(check_star)))
        print(np.nanstd(check_star))
    plt.legend()
    ax.set_xlabel('number')
    ax.set_ylabel('Diff Mag(mag)')
    ax.set_title('Light Curve:%s'%(args.wave_band[0]))
    print(args.wave_band)
    ax.invert_yaxis()
    plt.show()

    note = 'LC'

    return args, star, note

def sigmaclipmeanstd(dat):
    datc = stats.sigma_clip(dat, masked=False)
    return np.mean(datc), np.std(datc)
def c_nan(list):
    nan = [i for i in list if np.isnan(i)==True]
    r = len(nan)/len(list)
    return r

def find_refstars(matmag_cali):
    mag_me = [np.nanmean(i) for i in matmag_cali]
    tmp = list(map(list, zip(range(len(mag_me)), mag_me)))
    small = sorted(tmp, key=lambda x:x[1], reverse=False)
    values = []
    for i,j in enumerate(small[:40]):
        for k,l in enumerate(small[:40]):
            if i != k:
                chi =matmag_cali[j[0]] - matmag_cali[l[0]]
                std = np.nanstd(chi)
                if std < 0.0160:
                    value = [i, k, std]
                    if np.isnan(value[2]) == False and value[2] != 0:
                        values.append(value)
    v = [i[2] for i in values]
    tmp_v= list(map(list, zip(range(len(values)), v)))
    small_v = sorted(tmp_v, key=lambda x:x[1], reverse=False)
    s = [values[i[0]] for i in small_v][:100]
    count = [i[0] for i in s]
    print('Index of matmag_cali')
    print(s)
    star_ref = Counter(count)
    star_ref1 = []
    for i,j in star_ref.items():
        star_ref1.append(i)
    if len(star_ref1) < 10:
        star_ref1 = star_ref1[:5]
    else:
        star_ref1 = star_ref1[:8]
    print('Index of ref_stars')
    print(star_ref1)
    return star_ref1, small
    
