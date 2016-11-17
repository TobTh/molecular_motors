import numpy as np
import pandas
import math
import scipy as sci
import scipy.stats 
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from Measurement import LinFit

concentrations = np.array([5.0, 10.0, 20.0, 50.0, 100.0, 400.0, 1000.0, 2000.0])
pixellength = 1.57

pixels = np.array([7.0 , 17.18, 21.90, 88.71, 133.73, 170.68, 170.44, 134.93])
speeds  = pixels  * pixellength/10 
pixel_distance_errors = np.array([1.27, 3.92, 3.60, 6.07, 15.23, 13.39, 21.25, 14.15]) + 2*pixellength
speed_errors = pixel_distance_errors * pixellength/10
real_errors = speed_errors 

plotsize_x = 9
plotsize_y = 1/1.667 * plotsize_x

v_of_s = lambda S, vmax, Km: (vmax * S)/(Km + S)
linfit = lambda x, m, t: (x*m + t)

def fit_formula(x_es, y_es, function):
        (popt, pcov) = scipy.optimize.curve_fit(function, x_es, y_es)
        return(popt, np.sqrt(np.diag(pcov)))

def print_two_values(names, values, errors):
    print_error(names[0], values[0], errors[0])
    print("\n"),
    print_error(names[1], values[1], errors[1])
    print("\n"),
    print("\n"),

def print_error(name, value, error):
    print(str(name)+" = " + str(value)+" +- "+str(error)),
    
def get_fitted_arrays(fit_params, func, arange):
    new_func = lambda S: func(S, fit_params[0], fit_params[1])
    x_es = arange
    y_es = map(new_func, x_es)
    return(x_es, y_es)
        
def main():
    v_of_s = lambda S, vmax, Km: (vmax * S)/(Km + S)

    names = ("vmax", "Km")
    (all_values, all_errors) = fit_formula(concentrations, speeds, v_of_s)
    print_two_values(names, all_values, all_errors)

    (but_one_values, but_one_errors) = fit_formula(np.delete(concentrations, -1), np.delete(speeds, -1), v_of_s)
    print_two_values(("vmax but one", "Km but one"), but_one_values, but_one_errors)
    
    (x_es , y_es) = get_fitted_arrays(all_values, v_of_s, np.linspace(-10, 2150))
    (x_es_but_one, y_es_but_one) = get_fitted_arrays(but_one_values, v_of_s, np.linspace(-10, 2150))
    
    # Plot plot with two fits  
    figure = plt.figure()
    figure.set_size_inches(plotsize_x,plotsize_y)
    axis = figure.add_subplot(111)
    plt.xlim(-50, 2150)
    minorLocator1 = AutoMinorLocator()
    axis.plot(x_es, y_es , 'r-', label = "Fit mit allen Werten")
    axis.plot(x_es_but_one, y_es_but_one , 'g-', label = "Fit ohne letzten Wert")
    axis.yaxis.set_minor_locator(minorLocator1)
    axis.errorbar(concentrations, speeds, xerr = concentrations * 0.05, yerr =  real_errors, fmt = 'b^', label = "Konzentration in mmol")
    axis.legend(loc="best")
    axis.yaxis.grid(True, which='major')    
    axis.xaxis.grid(True, which='major')    
    plt.xlabel(r"Konzentration in mMol")
    plt.ylabel(r"$v$ in $\frac{\mu m}{s}$")
    figure.savefig("../bilder/both_fits.", bbox_inches='tight')

    x = 1/concentrations
    y = 1/speeds
    (values, errors) = fit_formula(x, y, linfit)
    print_two_values(("Km/vmax", "1/vmax"), values, errors)

    m = values[0]
    m_err = errors[0]
    t = values[1]
    t_err = errors[1]

    Km = m/t
    Km_error = math.sqrt((m_err/t)**2 + (m*t_err/t**2)**2)
    print_error("Km", Km, Km_error)
    print("\n"),

    vmax = 1/t
    vmax_err = t_err/t**2
    print_error("vmax", vmax, vmax_err)
    print("\n")
    
    (but_one_values, but_one_errors) = fit_formula(np.delete(x, -1), np.delete(y, -1), linfit)
    print_two_values(("Km/vmax without last","1/vmax without last"), but_one_values, but_one_errors)
    
    (x_es , y_es) = get_fitted_arrays(values, linfit, np.linspace(-0.025, 0.25))
    (x_es_but_one, y_es_but_one) = get_fitted_arrays(but_one_values, linfit, np.linspace(-0.025,0.25))

    # Plot plot with two fits  
    figure = plt.figure()
    figure.set_size_inches(plotsize_x,plotsize_y)
    axis = figure.add_subplot(111)
    minorLocator1 = AutoMinorLocator()
    axis.plot(x_es, y_es , 'r-', label = "Fit mit allen Werten")
    axis.plot(x_es_but_one, y_es_but_one , 'g-', label = "Fit ohne letzten Wert")
    axis.yaxis.set_minor_locator(minorLocator1)
    axis.errorbar(x, y, fmt = 'b^', label = r"$\frac{1}{S}$")
    axis.legend(loc="best")
    axis.yaxis.grid(True, which='major')    
    axis.xaxis.grid(True, which='major')    
    plt.xlabel(r"Konzentration in mMol")
    plt.ylabel(r"$v$ in $\frac{1}{v}$")
    figure.savefig("../bilder/both_fits_1over.", bbox_inches='tight')
       
    
if __name__ == "__main__":
    main()
