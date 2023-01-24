#!/usr/bin/python3
''' Code to plot Grotrian diagram of a given species in a given degree of ionization.
'''
#Modules import
import re
import sys
import time
import numpy as np 
import matplotlib.pyplot as plt

ifn = sys.argv[1]

#Input parameters
FIG_SIZE = (10, 5)
write_wl = True
write_title = False
reverse_baw = True
ext = 'png'

#Spectroscopic line of interest used if activate_loi is True
activate_loi = False
loi = {'Na 1': [5895.9242, 5889.9487], 
       'Th 2': [3741.183, 4019.1286, 4086.5203, 4094.7469, 4116.7133, 4250.3419, 4381.4009, 4381.8595, 5989.045],
       'K 1': [5801.7490, 6938.7630, 7664.8990, 7698.9643, 11769.6725, 12432.2730, 12522.1340]}

with open(ifn, 'r') as fd:
    content = fd.read().splitlines()
fd.close()

def parent_cfg_wo_term(cfg):
    ''' Remove terms in parenthesis inside the configuration
         3d7.(4F).4p          => 3d7.4p
         3d6.(5D).4s.4p.(1P*) => 3d6.4s.4p
         3d6.4s.(6D<7/2>).4f  => 3d6.4s.4f
    '''
    pcfg = re.sub('\.[\(<][a-zA-Z0-9\*<>/]*[\)>\.|\)$]', '', cfg)
    return pcfg

def term2num(term):
    ''' Convert spectral term into corresponding number
    '''
    L = ['S', 'S*', 'P', 'P*', 'D', 'D*', 'F', 'F*', 'G', 'G*', 'H', 'H*', 'I', 'I*', 'K', 'K*', 'L', 'L*', 'M', 'M*', 'N', 'N*', 'O', 'O*', 'Q', 'Q*', 'R', 'R*', 'T', 'T*', 'U', 'U*', '[', '(', '*', '']
    if term:
        if len(term) == 1 or term[1] == '*' or term[0] == '(':
            t = term[0]
        else:
            t = term[1:]
        if t in L: 
            return L.index(t)+1 
        else:
            return 0
    else:
        return L.index('')+1

def term2mult(term):
    ''' Convert spectral term into corresponding multiplicity
    '''
    mult_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9',\
                 'S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U',\
                 '1[','2[', '3[', '4[', '5[', '6[', '7[', '8[', '9[', '[', '(', '*', '']
    if term[0:2] in mult_list:
        return mult_list.index(term[0:2]) + 1
    elif term[0] in mult_list:
        return mult_list.index(term[0]) + 1 
    else:
        return 0

#Read element
elt = content[0].split(' ')[0]

print(f'ATOM NAME: {elt}')

#Find number of degree of ionization
degrees = []
start = False
for record in content:
    if start:
        if ' F ' in record: #End of levels
            break
        #match = re.match(r"^ *[0-9.]+ *[0-9.]+ *'*[0-9a-zA-Z.\(\)\<\>\/\*_]+ *[a-zA-Z]*[0-9A-Z.\[\]\*\/]+ *' * ([123]) *[0-9]+$", record)
        match = re.match(r"^ *[0-9.]+ *[0-9.]+ *'.*' * ([123]) *[0-9]+$", record)
        if match:
            degrees.append(int(match.groups()[0]))
    else:
        if 'NK' in record: #Start of levels
            start = True

degrees = sorted(list(set(degrees)))

print(f'IONIZATION DEGREES: {degrees}')


def find_ionization_index(content, degree):
    for record in content:
        match = re.match(f"^ *[0-9.]+ *[0-9.]+ *'.*' *{degree} *([0-9]+) *$", record)
        if match:
            idata = match.groups()
            return int(idata[0])

my_idx_ion_list = []
for degree in degrees:
    i = find_ionization_index(content, degree)
    my_idx_ion_list.append(i)


def read_el(content, degree):

    my_el, my_g, my_cfg, my_term, my_nk = [], [], [], [], []
    
    #Read level energies
    start = False
    for record in content:
        if start:
            if ' F ' in record:
                break
            #match = re.match(f"^ *([0-9.]+) *([0-9.]+) *'*([0-9a-zA-Z.\(\)\<\>\/\*_]+) *([a-zA-Z]?([0-9A-Z.\[\]\*\/]+ *)+) *' *{degree} *([0-9]+) *$", record)
            match = re.match(f"^ *([0-9.]+) *([0-9.]+) *'*([0-9a-zA-Z.\(\)\<\>\/\*_\+]+) *(.*) *' *{degree} *([0-9]+) *$", record)
            if match:
                idata = match.groups()
                #print(idata)
                my_el.append(float(float(idata[0])))
                my_g.append(int(float(idata[1])))
                my_cfg.append(parent_cfg_wo_term(idata[2]))
                my_term.append(re.sub(r'[a-z]', '', idata[3].strip()))
                my_nk.append(int(idata[4]))
            else:
                #print(f'EXCLUDED LEVEL: {record}')
                pass
        else:
            if 'NK' in record:
                start = True

    return my_el, my_g, my_cfg, my_term, my_nk

my_el_list, my_g_list, my_cfg_list, my_term_list, my_nk_list = [], [], [], [], []
term_all_list = []
for degree in degrees:
    my_el, my_g, my_cfg, my_term, my_nk = read_el(content, degree)
    my_el_list.append(my_el)
    my_g_list.append(my_g)
    my_cfg_list.append(my_cfg)
    my_term_list.append(my_term)
    my_nk_list.append(my_nk) #Level idx in the full atom
    #Split term for superatom
    term_all = []
    for term in my_term:
        for termi in term.split():
            term_all.append(termi)
    term_all_list.append(term_all)
    print(f'DEGREE {degree}: NUMBER OF ENERGY LEVELS: {len(my_el)}')

for degree, my_idx_ion, my_el in zip(degrees, my_idx_ion_list, my_el_list):
    print(f'INDEX FOR IONIZATION DEGREE {degree}: {my_idx_ion:3d} {my_el[0]:14.4f}')

def fancy_term(terms):
        my_fancy_term = []
        for term in terms:
            match_LS = re.match(r'^([1-9])([A-Z])(.*)$', term)
            match_JK = re.match(r'^([1-9])(\[[0-9]+/[0-9]\])(.*)$', term) 
            if match_LS:
                idata = match_LS.groups()
                mult = f'$^{idata[0]}$'
                L = idata[1]
                if idata[2] == '*':
                    parity = '$^o$'
                elif idata[2] in '123456789': #Case like '2D2' like in ThII
                    parity = idata[2]
                else:
                    parity = ''
                fancy_term = f'{mult}{L}{parity}'
            elif match_JK:
                idata = match_JK.groups()
                mult = f'$^{idata[0]}$'
                K = idata[1]
                if idata[2] == '*':
                    parity = '$^o$'
                else:
                    parity = ''
                fancy_term = f'{mult}{K}{parity}'
            else:
                fancy_term = term
            my_fancy_term.append(fancy_term)
        #print(f'DEGREE {degree}: FANCY  TERMS: {my_fancy_term}')
        return my_fancy_term

my_sorted_term_list, my_fancy_term_list = [], []
sorted_term_all_list, fancy_term_all_list = [], []
for degree, my_term, term_all in zip(degrees, my_term_list, term_all_list):
    #Sorted terms
    my_sorted_term = sorted(set(my_term), key=term2num)
    my_sorted_term = sorted(my_sorted_term, key=term2mult)
    my_sorted_term = sorted(my_sorted_term, key=lambda t: int(t.split('[')[1].split('/')[0]) if '[' in t else 0) #To sort  JK terms
    my_sorted_term = sorted(my_sorted_term, key=lambda t: int(t.split('(')[1].split(',')[1].split('/')[0]) if '(' in t else 0) #To sort  (j,j) terms
    my_sorted_term = sorted(my_sorted_term, key=lambda t: int(t.split('(')[1].split(',')[0]) if '(' in t else 0) #To sort  (j,j) terms


    my_sorted_term_list.append(my_sorted_term)
    term_all = sorted(set(term_all), key=term2num)
    term_all = sorted(term_all, key=term2mult)
    term_all = sorted(term_all, key=lambda t: int(t.split('[')[1].split('/')[0]) if '[' in t else 0) #To sort  JK terms
    term_all = sorted(my_sorted_term, key=lambda t: int(t.split('(')[1].split(',')[1].split('/')[0]) if '(' in t else 0) #To sort  (j,j) terms
    term_all = sorted(term_all, key=lambda t: int(t.split('(')[1].split(',')[0]) if '(' in t else 0) #To sort  (j,j) terms

    sorted_term_all_list.append(term_all)
    #print(f'DEGREE {degree}: TERMS:        {my_term}')
    if len(my_term) != len(term_all):
        print(f'DEGREE {degree}: ALL TERMS:    {term_all}')
    print(f'DEGREE {degree}: SORTED TERMS: {my_sorted_term}')
    #Fancy terms
    my_fancy_term_list.append(fancy_term(my_sorted_term))
    fancy_term_all_list.append(fancy_term(term_all)) #for superatom


def read_rt(content, degree, my_nk):
    #Read radiative bound-bound transitions
    j_el, i_el, fij, wl = [], [], [], []
    start = False
    for record in content:
        if start:
            if ' B-F ' in record:
                break
            # 62   1  2.704e-05  14 13.8  1.50 0  5.012e+07   -7.810  7.244e-07     3153.6970     33
            #357 355  1.288e-03  26 299.6 14.00 0  5.495e+08   -7.260  3.388e-05   114433.0800  15313
            #    9    1  1.014E-10   17     18.0   1.50 0  9.660E-01  220.300  5.369E-07      4993.281      8       5275.799
            #    7    1  3.258e-02  1000  10.2 10.20  1  1.326e+08     2.500  0.000e+00     4094.7469      1    4094.7470 'd' 's' loggf = -0.885 NZL
            match = re.match(r"^ *([0-9]+) *([0-9]+) *([0-9.eE\+\-]+) *([0-9]+) *([0-9.]+) *([0-9.]+) *[0-9]+ *([0-9.eE\+\-]+) *([0-9.\-]+) *([0-9.eE\+\-]+) *([0-9.]+) *([0-9]+) *([0-9.]+) .*$", record)
            if match:
                idata = match.groups()
                if int(idata[1]) in my_nk:
                    j_el.append(int(idata[0]))
                    i_el.append(int(idata[1]))
                    fij.append(float(idata[2]))
                    wl.append(float(idata[11]))
            else:
                print(f'RADIATIVE TRANSITION EXCLUDED: {record}')
        else:
            if ' F ' in record:
                start =  True
    return j_el, i_el, fij, wl

my_i_list, my_j_list, my_fij_list, my_wl_list = [], [], [], []
for degree, my_nk in zip(degrees, my_nk_list):
    my_j, my_i, my_fij, my_wl = read_rt(content, degree, my_nk)
    my_i_list.append(my_i)
    my_j_list.append(my_j)
    my_fij_list.append(my_fij)
    my_wl_list.append(my_wl)
    print(f'DEGREE {degree}: NUMBER OF RADIATIVE BBT: {len(my_wl)}')

def read_px(content, degree, my_nk):
    #Read radiative bound-free transitions
    px = []
    start = False
    for record in content:
        if start:
            if 'GENCOL' in record:
                break
            #     837 475 2.376e-19 11 911.0
            match = re.match(r'^ *([0-9]+) *([0-9]+) *([0-9.e\+\-]+) *[0-9]+ *[0-9.\-]+', record)
            if match:
                idata = match.groups()
                if int(idata[1]) in my_nk:
                    px.append(float(idata[2]))
        else:
            if ' A0 ' in record:
                start = True
    return px

my_px_list = []
for degree, my_nk in zip(degrees, my_nk_list):
    my_px = read_px(content, degree, my_nk)
    my_px_list.append(my_px)
    print(f'DEGREE {degree}: NUMBER OF RADIATIVE BFT: {len(my_px)}')

#Selection of the plot style
if reverse_baw:
    style = 'dark_background'
    color = 'w'
else:
    style ='default'
    color = 'k'


#plt.rc('text', usetex=True)
#plt.rc('font', size=22)

with plt.style.context(style):

    plt.figure(figsize=FIG_SIZE, dpi=300, facecolor='w', edgecolor='k', tight_layout=True) 
    for k, (degree, col) in enumerate(zip(degrees, [color, 'r'])):
        if degree == degrees[-1]:
            break
        
        gf = np.array(my_g_list[k])[np.array(my_i_list[k]) - my_idx_ion_list[k] - 1] * np.array(my_fij_list[k])
        loggf = np.log10(gf)
    
        print(f'DEGREE: {degree}: MIN F: {min(my_fij_list[k]):10.3e} MAX F: {max(my_fij_list[k]):10.3e} MIN GF: {min(gf):10.3e} MAX GF: {max(gf):10.3e} MIN LOGGF: {min(loggf)} MAX LOGGF: {max(loggf)}')
        
        plt.hist(loggf, bins=40, range=[-18, 2], histtype='step', label=f'{elt} {degree} $\\sum gf =${sum(10**loggf):10.3e}', color=col)
        #plt.hist(gf, bins=10000, range=[0, 10], histtype='step', label=f'{elt} {degree} $\\sum gf =${sum(gf):10.3e}', color=col)
    
    plt.xlabel('$\log{{gf}}$')
    #plt.xlabel('$gf$')
    #plt.xlim(0.0001, 10)
    plt.xlim(-18, 10)
    plt.ylim(1, 1e4)
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    if write_title:
        plt.title(ifn.replace('_','\_'))
    ofn0 = f'{ifn}_gf.{ext}'
    plt.savefig(ofn0)
    print(f'Control plot: {ofn0}')
    plt.close()

# Plot energy levels
for k, (degree, my_el, my_g, my_cfg, my_term, my_sorted_term, my_fancy_term, my_nk, my_i, my_j, my_fij, my_wl, my_px) in enumerate(zip(degrees, my_el_list, my_g_list, my_cfg_list, my_term_list, my_sorted_term_list, my_fancy_term_list, my_nk_list, my_i_list, my_j_list, my_fij_list, my_wl_list, my_px_list)):


    if degree == degrees[-1]:
        break

    with plt.style.context(style):

        fig = plt.figure(figsize=FIG_SIZE, dpi=300, facecolor='w', edgecolor='k', tight_layout=True)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
    
        #plt.ylabel('Energy [eV]', fontsize=12)
        ax1.set_ylabel('Energy [cm$^{-1}$]')
    
        #ax1.set_xticks(np.arange(len(my_fancy_term))+1)
        #ax1.set_xticklabels(my_fancy_term, rotation=65)
        ax1.set_xticks(np.arange(len(fancy_term_all_list[k]))+1)
        ax1.set_xticklabels(fancy_term_all_list[k], rotation=65)
    
        #Plot levels
        gmax = max(my_g)
        for el, g, cfg, term, nk in zip(my_el, my_g, my_cfg, my_term, my_nk):
            if len(term.split()) > 1:
                #when several spectral type are combined (for superatom)
                x_list, y_list = [], []
                for termi in term.split():
                    x_list.append(sorted_term_all_list[k].index(termi)+1)
                    y_list.append(el - my_el[0])
                x_list = sorted(x_list)
                el_line = ax1.scatter(x_list, y_list, marker='s', color=color, linewidths=1, alpha=(g/gmax)**0.5)
                el_line = ax1.plot(x_list, y_list, color, ls='dotted', alpha=0.3)
            else:
                el_line = ax1.plot(sorted_term_all_list[k].index(term)+1, el - my_el[0], f'{color}_', markersize=12, alpha=(g/gmax)**0.5)
            #el_line = ax1.plot(my_sorted_term.index(term)+1, el - my_el[0], 'k_', markersize=12, alpha=g/gmax)
            #plt.text(my_sorted_term.index(term)+1, el, cfg, fontsize=6)        
    
        #plot ionization
        ax1.axhline(y=my_el_list[k+1][0] - my_el[0], color=color, ls='dotted', alpha=0.5)
        ax1.axhline(y=0, color=color, ls='dotted', alpha=0.5)
    
        #Plot secondary y-axis (Energy in eV) 
        ax1ylim = np.array(ax1.get_ylim())
        ax2.set_ylim(6.61e-34*2.99e8/1.61e-19*ax1ylim*100)
        ax2.set_ylabel('Energy [eV]')
        if write_title:
            plt.title(ifn.replace('_', '\_'))
    
        ofn1 = f'{ifn}_el{degree}.{ext}'
        plt.savefig(ofn1)
        print(f'Control plot: {ofn1}')
    


        if not activate_loi:
            #Plot radiative bf transitions
            for el, term, nk, px in zip(my_el, my_term, my_nk, my_px):
                logr = np.log10(px/max(my_px))
                if logr < -10:
                    alpha = 0
                elif (logr >= -10) and (logr <0):
                    alpha = 1 - 0.1*abs(logr)
                else:
                    alpha = 1
                if len(term.split()) > 1:
                    #when several spectral type are combined (for superatom)
                    x_list, y_list = [], []
                    for termi in term.split():
                        x_list.append(sorted_term_all_list[k].index(termi)+1)
                    x = 2*[(max(x_list)+min(x_list))/2]
                    y = [el - my_el[0], my_el_list[k+1][0] - my_el[0]]
                    px_line = ax1.plot(x, y, 'r-', alpha=alpha**8)
                else:
                    #px_line = ax1.plot(2*[sorted_term_all_list[k].index(term)+1], [el - my_el[0], my_el_list[k+1][0] - my_el[0]], 'r-', alpha=alpha**8)
                    #px_line = ax1.plot(2*[my_sorted_term.index(term)+1], [el - my_el[0], my_el_list[k+1][0] - my_el[0]], 'r-', alpha=alpha**8)
                    px_line = ax1.plot(2*[sorted_term_all_list[k].index(term)+1], [el - my_el[0], my_el_list[k+1][0] - my_el[0]], 'r-', alpha=alpha**8, lw=0.5)
            
            leg1 = ax1.legend([f'{len(my_el)} levels \n $\Delta E_{{\min}} = {min(np.diff(my_el)):<10.3f}$ cm$^{{-1}}$', f'{len(my_wl):<6d} RBBT'], title=f'{elt} {degree}', loc=4)
            
            plt.setp(leg1.get_title(), fontsize=25)
            if write_title:
                plt.title(ifn.replace('_', '\_'))
            ofn3 = f'{ifn}_px{degree}.{ext}'
            plt.savefig(ofn3)
            print(f'Control plot: {ofn3}')
    
        #Plot radiative bb transitions

        for i, j, fij, wl in zip(my_i[:], my_j[:], my_fij[:], my_wl[:]):
            #print(len(my_sorted_term), len(my_term), len(my_el), i, j)
            try:
                loggf = np.log10(my_g[i-1-my_nk[0]]*fij)
            except IndexError:
                print(i, j)
                quit(1)
    
            if loggf < -5:
                alpha = 0
            elif (loggf >= -5) and (loggf < 0):
                alpha = 1 - 0.18*abs(loggf)
            else:
                alpha = 1
    
            #print(loggf, alpha)
            #alpha=1
            #rt_line = ax1.plot([my_sorted_term.index(my_term[i-1 - my_nk[0]])+1, my_sorted_term.index(my_term[j-1 - my_nk[0]])+1], [my_el[i-1 - my_nk[0]], my_el[j-1 - my_nk[0]]], 'r-', alpha=alpha**40)
            #rt_line = ax1.plot([my_sorted_term.index(my_term[i-1])+1, my_sorted_term.index(my_term[j-1])+1], [my_el[i-1] - my_el[0], my_el[j-1] - my_el[0]], 'r-', alpha=alpha**40)
    
            #when several spectral type are combined (for superatom)
            xi_list, xj_list = [], []
            iterm = my_term[i - my_idx_ion_list[k] - 1]
            jterm = my_term[j - my_idx_ion_list[k] - 1]
            if len(iterm.split()) > 1:
                xi_list = [sorted_term_all_list[k].index(termi) + 1 for termi in iterm.split()]
                x1 = (max(xi_list) + min(xi_list)) / 2
            else:
                x1 = sorted_term_all_list[k].index(my_term[i - my_idx_ion_list[k]]) + 1
            if len(jterm.split()) > 1:
                try:
                    xj_list = [sorted_term_all_list[k].index(termj) + 1 for termj in jterm.split()]
                except ValueError:
                    print(jterm)
                x2 = (max(xj_list) + min(xj_list)) / 2
            else:
                #x2 = sorted_term_all_list[k].index(my_term[j - my_idx_ion_list[k]]) + 1     
                x2 = sorted_term_all_list[k].index(my_term[j - my_idx_ion_list[k] - 1]) + 1     
    
            #print(my_term[i - my_idx_ion_list[k]], my_term[j - my_idx_ion_list[k]])
    
            y1 = my_el[i - my_idx_ion_list[k] ] - my_el[0]
            #y2 = my_el[j - my_idx_ion_list[k] ] - my_el[0]
            y2 = my_el[j - my_idx_ion_list[k] - 1] - my_el[0]
            
            if write_wl and activate_loi and (abs(wl - np.array(loi[f'{elt} {degree}'])) < 5e-3).any():
                rt_line = ax1.plot([x1, x2], [y1, y2], alpha=alpha, lw=0.5, label=f'{wl:>9.3f} \u00C5, log($gf$) = {loggf:6.3f}')
                print(f'{wl:9.3f} \u00C5, log(gf) = {loggf:6.3f}, alpha = {alpha:5.3f}')
            else:
                if not activate_loi:
                    rt_line = ax1.plot([x1, x2], [y1, y2], 'r-', alpha=alpha, lw=0.5)
        
        if write_wl:
            ax1.legend(loc=4)
    
        if not write_wl:
            #Plot levels
            gmax = max(my_g)
            for el, g, cfg, term, nk in zip(my_el, my_g, my_cfg, my_term, my_nk):
                if len(term.split()) > 1:
                    #when several spectral type are combined (for superatom)
                    x_list, y_list = [], []
                    for termi in term.split():
                        x_list.append(sorted_term_all_list[k].index(termi)+1)
                        y_list.append(el - my_el[0])
                    x_list = sorted(x_list)
                    el_line = ax1.scatter(x_list, y_list, marker='s', color=color, linewidths=1, alpha=(g/gmax)**0.5)
                    el_line = ax1.plot(x_list, y_list, color, ls='dotted', alpha=0.3)
                else:
                    el_line = ax1.plot(sorted_term_all_list[k].index(term)+1, el - my_el[0], f'{color}_', markersize=12, alpha=(g/gmax)**0.5)
                #el_line = ax1.plot(my_sorted_term.index(term)+1, el - my_el[0], 'k_', markersize=12, alpha=g/gmax)
                #plt.text(my_sorted_term.index(term)+1, el, cfg, fontsize=6)   
    
        if write_title:
            plt.title(ifn.replace('_', '\_'))
    
        ofn2 = f'{ifn}_rt{degree}.{ext}'
        plt.savefig(ofn2)
        print(f'Control plot: {ofn2}')
    
        #plt.show()


quit(1)
