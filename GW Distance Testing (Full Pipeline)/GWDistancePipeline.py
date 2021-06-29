import math
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
import scipy.ndimage as rnoise

import scipy.stats as stats
import time


# Create max strain array

def max_strain(strain_arr):
    max_str = [] 
    for i in np.arange(len(strain_arr)):
        max_str.append(max(abs(strain_arr[i])))
    return max_str

# Create avg strain array

def avg_strain(strain_arr):
    avg_str = [] 
    for i in np.arange(len(strain_arr)):
        avg = abs(np.sum(strain_arr[i])/len(strain_arr[i]))
        avg_str.append(avg)
    return avg_str

# Hubble's Law

def vel(dist, H0):
    return H0*dist

# Guess Inc

def best_i(actual_dis, f): # a helper function for best_i_arr, does similar things as best_angle
    angle_lst = np.arange(0, math.pi , 0.001)
    best_dif = float('inf')
    best_i = 0
    for i in range(len(angle_lst)):
        d = f(angle_lst[i])
        
        #print('---')
        #print(str(angle_lst[i])+' | ' + str(d))
        if abs(d - actual_dis) < best_dif:
            best_i = angle_lst[i]
            best_dif = abs(d - actual_dis)
            
            
            #print(best_dif)
            #print('---')
    #print('Best i: ' + str(best_i))

    return best_i

# GW Distance Calc
def distance_to_GW(t, M_sol, h_max, i):
    '''This function will give the non-localized distance (in Mpc) to the gravitational wave 
    when inputting time in seconds, combined mass (or chirp mass) in solar masses, and the maximum strain in Hz.'''
    
    G = 6.67e-11 # N kg^-2 m^2
    c = 3e8 # m/s
    
    M = M_sol * (2 * 10**30) # gives mass in kg
    
    term1 = (5/256)**(3/8)
    term2 = ((c**3) / (G * M))**(5/8)
    term3 = (1 / (t**(3/8)))
    term4 = (t**(5/8))
    
    orbital_freq = term1 * term2 * term3
    orbital_phase = np.round(0.36571582 * term2 * term4) #round terms to third sig fig, round the constant to third sig fig
    
    distance = (2*c / h_max) * (G*M / (c**3))**(5/3) * orbital_freq**(2/3) *(1 + np.cos(i)**2)* abs(np.cos(2 * orbital_phase)) # this is distance in meters
    
    #print(orbital_freq) # printed this just to check the value of it
    #print(orbital_phase) # printed this just to check the value of it
    #print(orbital_freq)
    #print(orbital_phase)
    return distance / (9.223 * 10**18)#i returns distance in Mpc. 2.25 update: change [distance / (9.223 * 10**18), i]
    # distance only, for debugging best_i_arr. If anyone need to use best_angle again, please change the return statement to the original one



# guess n procedure 

def guess_n(lum_dist, times, mass, max_strain):
    # guess the n value 
    
    n_new = []
    d_old_arr = []
    for x in np.arange(len(times)):
        
        # Determine d_old and inc_old
        inc_old = best_i(lum_dist[x], lambda i: distance_to_GW(times[x], mass[x], max_strain[x],i)) # OG inc
        d_old = distance_to_GW(times[x], mass[x], max_strain[x],inc_old)
        
        # Determine the n_guess  
        n_guess = d_old/lum_dist[x]  # D values in Mpc
        if np.round(n_guess,4) == 0.9999:
            n_guess = np.round(n_guess) 
        elif np.round(n_guess,1) == 0.1: # could be a constraint on n 
            n_guess = 9 # try changing this to 1 or 9 and see what happens...9works best
        elif np.round(n_guess,1) != 0.1 and np.round(n_guess,4) != 0.9999: 
            n_guess = np.round(n_guess*10)-1
        
        #print('n_guess = ' + str(n_guess))
        n_new.append(n_guess)
        d_old_arr.append(d_old)
  
    return n_new

# Guess inc procedure 

def inc_calc(lum_dist, times, mass, max_strain, n_new):
    # use n_guess to create array of inc values to test     
    inc_arr = []
    k_rng = np.arange(0,2,0.001) # create range of k values to test  
    for x in np.arange(len(times)): 
        i_arr = []
        for k in k_rng:
            i = np.arccos(np.sqrt(abs((2*max_strain[x]*n_new[x])-1))) + (2*np.pi*k) # rads
            i_arr.append(np.round(i,3))      
        ind = np.where(np.array(i_arr) <= np.pi)
        ind = ind[0].astype(int)
        inc_arr.append(np.array(i_arr)[ind])    
    return inc_arr

# Get new distance and inclination angle for all target inc values 

def tgt_val_calc(lum_dist, times, mass, max_strain, n_new, inc_new): 
    # returns target distance and inc values for array of incs from inc_calc 
    dist_arr = []
    
    for x in np.arange(len(times)):
        incs = inc_new[x]
        dists = []
        
        for ind in np.arange(len(incs)):
            d = distance_to_GW(times[x], mass[x], max_strain[x],incs[ind])*n_new[x]
            dists.append(d)
            
        dist_arr.append(dists)
    return dist_arr

# filter dist and inc arrays to values that are close to d_actual

def target_acq(d_new, lum_dist, inc_new):
    d_tgt = []
    inc_tgt = []
    for x in np.arange(len(lum_dist)):
        target = np.isclose(d_new[x], lum_dist[x], atol = 0.2, rtol = 1e-2)
        target_inds = np.where(target == True)
        d_tgt.append(np.array(d_new[x])[target_inds[0]])
        inc_tgt.append(np.array(inc_new[x])[target_inds[0]])
    return d_tgt, inc_tgt

# return the error for each target dist and inc value

def error_calc(d_tgt, inc_tgt, lum_dist):
    error_arr = []
    for x in np.arange(len(lum_dist)):
        error = []
        for d in d_tgt[x]:
            err = abs(d - lum_dist[x])/lum_dist[x] * 100
            error.append(err)
        error_arr.append(error)
    return error_arr

def guess_results(error_arr, lum_dist, d_tgt, inc_tgt):
    restult_arr = []
    for x in np.arange(len(lum_dist)):
        if len(error_arr[x]) == 0:
            restult_arr.append(0)
        elif len(error_arr[x]) != 0:
            tgt_ind = np.where(error_arr[x] == min(error_arr[x]))
            tgt_ind = tgt_ind[0].astype(int)
            d = d_tgt[x][tgt_ind[0]]
            i = inc_tgt[x][tgt_ind[0]]
            restult_arr.append([d, i, min(error_arr[x])])
    return restult_arr


def Standard_Siren_caclulator(lum_dist, times, mass, max_strain, z, GW_names, sig_ligo, sig_z, plot=False, prt=False, inc_tsting=False):
    
    # Determine d_old 
    d_old = []
    
    # for inc testing 
    inc_old_val = []
    
    for x in np.arange(len(times)):
        # Calculate distance and inc using standard method 
        inc = best_i(lum_dist[x], lambda i: distance_to_GW(times[x], mass[x], max_strain[x],i)) # OG inc
        d = distance_to_GW(times[x], mass[x], max_strain[x],inc)
        d_old.append(d)
        inc_old_val.append(inc)
        
    # determine n_guess
    n_guess = guess_n(lum_dist, times, mass, max_strain)
    
    # determine inc_new
    inc_new = inc_calc(lum_dist, times, mass, max_strain, n_guess)
    
    # determine d_new
    d_new = tgt_val_calc(lum_dist, times, mass, max_strain, n_guess, inc_new)
    
    # Determine target distance values 
    target = target_acq(d_new, lum_dist, inc_new)
    
    # Determine the errors of d_new
    error_arr = error_calc(target[0], target[1], lum_dist)
    
    # Determine target values 
    r_check = guess_results(error_arr, lum_dist, target[0], target[1])
    
    # Find best target values by checking errors and results 
    
    distances = []
    d_old_arr = []
    d_act_arr = []
    d_sig_arr = []
    inclinations = []
    inc_new_arr = []
    errors = []
    GW_name_gd = []
    z_new = []
    z_sig_arr = []
    n_new = []
    
    #for inc testing
    d_new_dom = []
    d_new_val = []
    inc_new_val = []

    for x in np.arange(len(lum_dist)):
        results = np.array(r_check[x])
        
        if inc_tsting == True: 
            d_new_dom.append(d_new[x])
            
        if results.all() == 0: 
            # for inc testing
            d_new_val.append(0)
            inc_new_val.append(0)
            continue
        elif results.all() != 0:
            d = results[0]
            inc = results[1]
            err = results[2]
            distances.append(d)
            inclinations.append(inc)
            errors.append(err)
            GW_name_gd.append(GW_names[x])
            inc_new_arr.append(inc)
            z_new.append(z[x])
            n_new.append(n_guess[x])
            d_old_arr.append(d_old[x])
            d_act_arr.append(float(lum_dist[x]))
            d_sig_arr.append(sig_ligo[x])
            z_sig_arr.append(sig_z[x])
            
            #for inc testing
            d_new_val.append(d)
            inc_new_val.append(inc)
    
    # Find v
    n_sorted = [n_new[2], n_new[1], n_new[3], n_new[5], n_new[4], n_new[0]]
    z_sorted = [z_new[2], z_new[1], z_new[3], z_new[5], z_new[4], z_new[0]]
    c = 3e+5 # km/s
    v_sorted = np.array(z_sorted).astype(float)*c # km/s
    
    # sorting the arrays 
    GW_names_gd_sorted = [GW_name_gd[2], GW_name_gd[1], GW_name_gd[3], GW_name_gd[5], GW_name_gd[4], GW_name_gd[0]]
    errors_sorted = [errors[2], errors[1], errors[3], errors[5], errors[4], errors[0]]
    distances_sorted = sorted(distances)
    
    d_old_sorted = [d_old_arr[2], d_old_arr[1], d_old_arr[3], d_old_arr[5], d_old_arr[4], d_old_arr[0]]
    incs_sorted = [inc_new_arr[2], inc_new_arr[1], inc_new_arr[3], inc_new_arr[5], inc_new_arr[4], inc_new_arr[0]]
    d_act_sorted = sorted(d_act_arr)
    d_sig_sorted = [d_sig_arr[2], d_sig_arr[1], d_sig_arr[3], d_sig_arr[5], d_sig_arr[4], d_sig_arr[0]]
    
    z_sig_sorted = [z_sig_arr[2], z_sig_arr[1], z_sig_arr[3], z_sig_arr[5], z_sig_arr[4], z_sig_arr[0]]
    
    # Calculate H0 for each GW (H0_i)
    
    H0_i = v_sorted/distances_sorted
    
    # Determine the uncertanity on D (sig_d)
    num_d = np.array(d_old_sorted)
    denom_d = np.array(d_act_sorted)
    sig_d_i = np.array(d_sig_sorted)
    sig_d = np.sqrt(((-(num_d/denom_d)**2)**2) * (np.array(sig_d_i)**2))
    
    # Determine the Uncertanity in vel (sig_v)
    part_v = c
    # sig_v = np.sqrt((part_v**2) * (np.array(z_sig_sorted)**2))
    sig_v = np.sqrt((c**2)*(np.array(z_sig_sorted)**2)) # sig_v when ignoring the constant
    
    # Determine the uncertanity on H_i (sig_hi)
    phv = 1/np.array(distances_sorted)
    phd = - np.array(v_sorted) / (np.array(distances_sorted)**2)
    
    sig_hi = np.sqrt(((phv**2)*(sig_v**2)) + ((phd**2)*(sig_d**2)))
    
    # Finding Hbar and its uncertanity including Kilonova
    hb_num = np.sum(np.array(H0_i)/np.array(sig_hi)**2)
    hb_denom = np.sum(1/(np.array(sig_hi)**2))
    
    H_bar = np.sum(np.array(H0_i)/np.array(sig_hi)**2) / np.sum(1/np.array(sig_hi)**2)
    #H_bar = hb_num/hb_denom
    
    sig_Hbar = np.sqrt(1/np.sum(1/np.array(sig_hi)**2)) # technically sig_Hbar_sqrt... uncertanity = np.sqrt(sig_Hbar)
    
    
    # Finding Hbar and its uncertanity without  Kilonova (added 4/16/2021)
    hb_num_min = np.sum(np.array(H0_i[1:])/np.array(sig_hi[1:])**2)
    hb_denom_min = np.sum(1/(np.array(sig_hi[1:])**2))
    
    H_bar_min = np.sum(np.array(H0_i[1:])/np.array(sig_hi[1:])**2) / np.sum(1/np.array(sig_hi[1:])**2)
    
    sig_Hbar_min = np.sqrt(1/np.sum(1/np.array(sig_hi[1:])**2)) # technically sig_Hbar_sqrt... uncertanity = np.sqrt(sig_Hbar)
    
    # Plot Results 
    
    #Find the fitted line
    xfit = np.linspace(distances_sorted[0], distances_sorted[-1], num=len(distances_sorted))
    coeffs = poly.polyfit(distances_sorted[:4],v_sorted[:4], 1)
    yfit = poly.polyval(xfit,coeffs)  
    
    if plot == True: 
        plt.figure(figsize=(8,5))
        plt.loglog(xfit, yfit, label = 'Fitted Line (Function)', color = 'k', linestyle = '--')
        plt.errorbar(distances_sorted, v_sorted, xerr = sig_d, yerr = sig_v, 
                     color = 'r', label = 'Data w/ Error Bars', fmt = 'o')
        plt.title('$v = H_0 d_L$')
        plt.ylabel('Velocity [km/s]')
        plt.xlabel('Distance [Mpc]')
        plt.legend()
        plt.grid()
        plt.show()
        # add final results to legend of plot
    
    
    if prt == True:
        print('Results for each GW')
        for x in np.arange(len(distances_sorted)):
            print(GW_names_gd_sorted[x])
            print('-------------------')
            print('z ≈ ' + str(z_sorted[x]))
            print('n ≈ ' + str(int(n_sorted[x])))
            print('D_guess ~ ' + str(np.round(distances_sorted[x], 3)) + ' ± ' + str(float(sig_d_i[x])) + ' Mpc')
            print('i_guess ≈ ' + str(incs_sorted[x]) + ' rad, ' + str(np.round(np.rad2deg(incs_sorted[x]))) + '°') 
            print('H0_estimate ≈ ' + str(np.round(H0_i[x],3)) +' ± ' +
                  str(np.round(sig_hi[x],3)) +' km/s/Mpc') # errors need to be fixed
            print('')
        print('--')    
        print(' ')
        print('Final Results')
        print('------------------------')
        print('H0 including GW170817: H0 ~ ' + str((np.round(H_bar,3))) + ' ± ' + str(np.round(sig_Hbar,4))  + ' km/s/Mpc')
        print('H0 excluding GW170817: H0 ~ ' + str((np.round(H_bar_min,3))) + ' ± ' + str(np.round(sig_Hbar_min,4))  + ' km/s/Mpc')
        #print('Absolute Error for H0_actual ~ ' + str(np.round(abs(H_bar-(70.3))/(70.3) *100, 3)) + '%')
        #print('Absolute Error For H0_actual upper Limit~ ' + 
        #      str(np.round(abs(H_bar-(70.3+5.3))/(70.3+5.3) *100, 3)) + '%') # errors on line above need fix
    
    if inc_tsting == True:
        print(' ')
        print('Returns: d_new_dom, d_old, inc_old_val, d_new_val, inc_new_val, n_guess')
        return d_new_dom, d_old, inc_old_val, d_new_val, inc_new_val, n_guess 
    else: 
    #if inc_tsting == False: 
        print(' ')
        print('Returns: H_bar, sig_Hbar, H_bar_min, sig_Hbar_min, n_new')
        return H_bar, sig_Hbar, H_bar_min, sig_Hbar_min, n_new
    
    #return r_check, z_sorted,GW_names_gd_sorted, distances_sorted, H_bar_min, sig_Hbar_min
    #return H_bar, sig_Hbar
    #return H0_i, sig_hi, distances_sorted, sig_d, v_sorted, sig_v, yfit 
    #return H_bar, sig_Hbar, H_bar_min, sig_Hbar_min, n_new


