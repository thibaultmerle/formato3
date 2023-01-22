#!/usr/bin/python3
#
# formato version 3
# 
# Build one ionization degree at a time. Use mergato.py for combining different ionization degrees
#
# Usage: formato.py valdfilename [outputfilename]
#
# Input format:
# http://vald.astro.uu.se/~vald/php/vald.php
# In "Extract element"
# 1 < λ < 1e6 Å (air)
# FTP
# Long format
# Set "Term designation"
#Linelist configuration: choose "custom" and check for duplicates
# Unit selection: Energy unit: 1/cm - Medium: air - Wavelength unit: angstrom - VdW syntax: extended

#Modules import
import re
import sys
import time
import numpy as np 
import scipy.special as ss 
import scipy.ndimage as sn
import scipy.optimize as so
import scipy.interpolate as si

#Parameters
#Energy level parameters
e_max = 12 # below 50: in eV; above 50: in cm⁻¹
#Broadening parameters
t_broad = 4500 #6000 # [K] Temperature for broadening parameter to estimate strength of line and then deduce the line frequency points
nh_broad = 1.e21 # 1.e23 # [m⁻³] Hydrogen number density to estimate strength of line and then deduce the line frequency points
ne_broad = 1.e17 # 1.e19 # [m⁻³] Free electron number density to estimate strength of line and then deduce the line frequency points
#Line frequency parameters
fcut = 1e-10 #Below this value, the number of frequency points is minimum
nq_min = 20 #Minimum number of frequency points
q0_min = 4.0 #Transition in Doppler width unit between linear and log scales
qmax_min = 5.0 #Minimum of maximum Doppler width unit
#Photoionization parameters
npi = 150 #Maximum number of photo-xsection desired
i_min = 2 #Number of initial xsection kept at the photoionization threshold
gaussian_width0 = 20 #Initial width of the Gaussian kernel for convolution
std_frac = 0.01 #Fraction of the xsection standard deviation to apply for OS
tol_max = 0.10 # Maximum tolerance on the relative error of the xsection area
resampling = True # Resampling of photoionization xsections
#Collision parameters
t_coll = [1000, 3000, 5000, 7000, 10000, 15000, 20000] #Temperature values for the collisions rates
sce_type = 'SEATON' #Type of electron collisions [VAN REGEMORTER | SEATON]
sh = 1.0 #Drawin enhancement factor for hydrogen collisions
#Line of interest parameters
activate_loi = True
nq_loi = 100 #Number of frequency points to use in conjonction with IOPACL=1 and ABSLIN for background line opacities in MULTI

#Running parameters
write_el = False #Display energy levels 
write_bp = False #Display broadening parameters
plot_freq = True #Display control plot for frequency plot selection

#Spectroscopic line of interest (Not mandatory for the building of the atom)
loi = {'Na 1': [5895.9242, 5889.9487], 
       'Th 2': [3741.183, 4019.1286, 4086.5203, 4094.7469, 4116.7133, 4250.3419, 4381.4009, 4381.8595, 5989.045],
       'K 1': [5801.7490, 6938.7630, 7664.8990, 7698.9643, 11769.6725, 12432.2730, 12522.1340]}

#Atomic data for ionization set manually (mandatory)
#Ionization energy 
ion_el   = {'Na 1': 41449.451,
            'Mg 1': 61671.05,
            'K 1': 35009.8140, 
            'Co 1': 63564.6, 'Co 2': 137795.,
            'Ni 1': 61619.77, 'Ni 2':146541.56,
            'Zn 1': 75769.31, 'Zn 2': 144892.6,
            'Ba 2': 80686.30,
            'Ce 1':  44672.,  'Ce 2': 88370.,
            'Th 2': 97600.} # cm⁻¹
#Statistical weight of the ionized level
ion_g    = {'Na 1': 1,
            'Mg 1': 2,
            'K 1': 1,
            'Co 1': 9, 'Co 2': 10,
            'Ni 1': 6, 'Ni 2': 21,
            'Zn 1': 2, 'Zn 2': 1,
            'Ba 2': 1,
            'Ce 1': 44, 'Ce 2': 33, #Check if we want the g from fine level or mean level
            'Th 2': 49 #Following Mashonkina et al. 2012 
            } 
#Electronic configuration of the ionized level
ion_cfg  = {'Na 1': '2p6',
            'Mg 1': '3s',
            'K 1': '3p6',
            'Co 1': '3d8', 'Co 2':'3d7',
            'Ni 1': '3d9', 'Ni 2': '3p6.3d8',
            'Zn 1': '3d10.4s', 'Zn 2': '3d10',
            'Ba 2': '5p6',
            'Ce 1': '4f.5d2', 'Ce 2': '4f2',
            'Th 2': '5f.6d' }
#Spectral term of the ionized level
ion_term = {'Na 1': '1S',
            'Mg 1': '2S',
            'K 1': '1S',
            'Co 1': 'a3F', 'Co 2': 'a4F',
            'Ni 1': '2D', 'Ni 2': '3F',
            'Zn 1': '2S', 'Zn 2': '1S',
            'Ba 2': '1S',
            'Ce 1': '4H', 'Ce 2': '3H',
            'Th 2': '3H*'}  
#From https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl in atomic mass unit (amu)
mass =     {'H 1': 1.008,
            'Na 1': 22.990,
            'Mg 1': 24.305,
            'K 1': 39.0983,
            'Co 1': 58.9332, 'Co 2': 58.9332,
            'Ni 1': 58.6934, 'Ni 2': 58.6934,
            'Zn 1': 65.409,  'Zn 2': 65.409,
            'Ba 2': 131.293,
            'Ce 1': 140.116, 'Ce 2':140.116,
            'Th 2': 232.038} 
#Abundances from Grevesse, Asplund & Sauval (2007) (e.g. visible here https://slideplayer.com/slide/7055595/)
abund =    {'Na 1': 6.17,
            'Mg 1': 7.53,
            'K 1': 5.08,
            'Co 1': 4.92, 'Co 2': 4.92,
            'Ni 1': 6.23, 'Ni 2': 6.23,
            'Zn 1': 4.60, 'Zn 2': 4.60,
            'Ba 2': 2.17,
            'Ce 1': 1.70, 'Ce 2': 1.70,
            'Th 2': 0.06}

norad_en =  {'Ni 2': 'ad/norad/ni/ii/ni2.en.ls.txt'}
norad_px =  {'Ni 2': 'ad/norad/ni/ii/ni2.ptpx.txt'}
norad_rrc = {'Ni 2': 'ad/norad/ni/ii/ni2.rrc.txt'}


#Fundamental constants
ry = 10973731.57      # m⁻¹  Rydberg constant
hp = 6.62606957e-34   # Js   Planck constant
kb = 1.3807e-23       # J/K  Boltsmann constant
cc = 2.99792458e8     # m/s  speed of light in vacuum
qe = 1.602176565e-19  # C    elementary charge
me = 9.10938215e-31   # kg   electron mass
amu = 1.660538921e-27 # kg   atomic mass unit

ifn = sys.argv[1] #name of the vald file 

#Set the model atom output filename
try:
    ofn = sys.argv[2]
except IndexError:
    ofn = None

#Uniformize the user parameter e_max (in eV or cm⁻¹) into e_lim (in cm⁻¹)
#and set maximum energy unit
if e_max < 50:
    e_lim = qe/hp/cc * e_max / 100 #convert eV into cm⁻¹
    e_max_unit = 'eV'
else:
    e_lim = e_max 
    e_max_unit = 'cm⁻¹'


#Read VALD file content
print(f'\nREAD INPUT: {ifn}')
with open(ifn, 'r') as fd:
    content = fd.read().splitlines()

#Extract VALD data
elt = []
valdline, leveli, levelj = [], [], []
ei, ej, gi, gj, cfgi, cfgj, termi, termj, loggf, wl, gr, gs, gv = [], [], [], [], [], [], [], [], [], [], [], [], []
loggf_ref = []

vald_nlin = 0
for k, record1 in enumerate(content[2::4]): #Each atomic line is made for 4 records starting at the 3rd line
    vald_nlin += 1
    exp = r"^'([a-zA-Z0-9 ]+)',([0-9. ]+),([0-9.\- ]+),([0-9. ]+),([0-9. ]+),([0-9. ]+),([0-9. ]+),[0-9.\- ]+,[0-9.\- ]+,[0-9.\- ]+,([0-9. -]+),([0-9. -]+),([0-9. -]+)$"
    match = re.match(exp, record1)   

    try:
        record2 = content[2 + 4*k+1] # Lower level (coupling + electronic configuration + term) 
        record3 = content[2 + 4*k+2] # Upper level (coupling + electronic configuration + term)
        record4 = content[2 + 4*k+3] # References
    except IndexError:
        break #End of file

    matchi = re.match(r"^' *([LSJK? ]+) *(.*) +(.*)'$", record2)
    matchj = re.match(r"^' *([LSJK? ]+) *(.*) +(.*)'$", record3)

    if matchi: # Match of low configuration (record2)
        idata = matchi.groups()
        #print(idata)
        if idata[1].strip() in ['-', '', 'unknown']:

            if idata[0].strip() == '': #Case where only configuration is given and is in idata[2] (like in ThII)
                if idata[2].strip()[-1] == '*':
                    cfgi.append(idata[2].strip()[:-1])
                    termi.append('*')
                else:
                    cfgi.append(idata[2].strip())
                    termi.append('')
            else:
                print(f'READ INPUT {k+1:5d}: LINE SKIPPED: CONFIGURATION FORMAT ISSUE: {record2}')
                continue

        elif idata[2].strip()[-1] == 'e':
            print(f'READ INPUT {k+1:5d}: LINE SKIPPED: SPECTRAL TERM FORMAT ISSUE: {record2}')
            continue
        else:
            cfgi.append(idata[1].strip().replace(' ', ''))
            termi.append(idata[2])
    else:
        print(f'READ INPUT {k+1:5d}: LINE SKIPPED: NO MATCH: {record2}')
        vald_nlin -= 1
        continue

    if matchj: # Match of up configuration (record3)
        idata = matchj.groups()
        #print(idata)
        #input()
        if idata[1].strip() in ['-', '', 'unknown']:

            if idata[0].strip() == '': #Case where only configuration is given and is in idata[2] (like in ThII)
                if idata[2].strip()[-1] == '*':
                    cfgj.append(idata[2].strip()[:-1])
                    termj.append('*')
                else:
                    cfgj.append(idata[2].strip())
                    termj.append('')
            else:
                print(f'READ INPUT {k+1:5d}: LINE SKIPPED: CONFIGURATION FORMAT ISSUE: {record3}')
                cfgi.pop()
                termi.pop()
                continue

        elif idata[2].strip()[-1] == 'e':
            print(f'READ INPUT {k+1:5d}: LINE SKIPPED: SPECTRAL TERM FORMAT ISSUE: {record3}')
            cfgi.pop()
            termi.pop()
            continue         
        else:
            cfgj.append(idata[1].strip().replace(' ', ''))
            termj.append(idata[2])
    else:
        print(f'READ INPUT {k+1:5d}: LINE SKIPPED: NO MATCH: {record3}')
        vald_nlin -= 1
        cfgi.pop()
        termi.pop()
        continue

    if match: # Match of atomic data  (record1)

        idata = match.groups()
        #print(f'{k} {idata}')

        #Skip lines involving levels with energy higher than e_max
        if (float(idata[3]) > e_lim) or (float(idata[5]) > e_lim):
            cfgi.pop(), termi.pop()
            cfgj.pop(), termj.pop()
            continue 

        elt.append(idata[0])
        valdline.append(record1)
        leveli.append(record2[1:-1])
        levelj.append(record3[1:-1])
        wl.append(float(idata[1]))
        loggf.append(float(idata[2]))
        ei.append(float(idata[3]))
        gi.append(int(2*float(idata[4]) + 1))
        ej.append(float(idata[5]))
        gj.append(int(2*float(idata[6]) + 1)) 
        gr.append(float(idata[7]))       
        gs.append(float(idata[8])) 
        gv.append(float(idata[9]))  

        #Collect the loggf reference
        matchref = re.match(r"^'.* gf:([a-zA-Z0-9]+) .*'$", record4) 
        
        if matchref:
            ref = matchref.groups()[0]
            loggf_ref.append(f'{loggf[-1]:7.3F} {ref}')

    else:
        print(record1)
        quit(1)

#Conversion into array
valdline = np.array(valdline)
leveli = np.array(leveli)
levelj = np.array(levelj)
ei = np.array(ei)
ej = np.array(ej)
gi = np.array(gi)
gj = np.array(gj)
cfgi = np.array(cfgi)
cfgj = np.array(cfgj)
termi = np.array(termi)
termj = np.array(termj)
loggf = np.array(loggf)
wl = np.array(wl)
gr = np.array(gr)
gs = np.array(gs)
gv = np.array(gv)
loggf_ref = np.array(loggf_ref)

print(f"DATA INPUT: VECTOR SIZES MUST BE IDENTICAL: {valdline.size} {leveli.size} {levelj.size} {ei.size} {ej.size} {gi.size} {gj.size} {cfgi.size} {cfgj.size} {termi.size} {termj.size} {loggf.size} {wl.size} {gr.size} {gs.size} {gv.size} {loggf_ref.size}")


if len(set(elt)) == 1:
    elt = list(set(elt))[0]
else:
    print(f'READ INPUT: ONLY ONE SPECIES AND ONE IONIZATION DEGREE AT A TIME: {set(elt)}')
    quit(1)

#Check the existence of the ionization data for a given species
try:
    ion_el[elt]
    ion_g[elt]
    ion_cfg[elt]
    ion_term[elt]
    mass[elt]
    abund[elt]
except KeyError:
    print(f"DATA INPUT: IONIZATION DATA SHOULD BE ADDED IN ion_el, ion_g, ion_cfg & ion_term FOR {elt}")
    quit(1)

if not ofn:
    ofn = f"atom.{elt.replace(' ', '')}"

print(f'\nDATA INPUT: ELEMENT:    {elt}')
print(f'DATA INPUT: NUMBER OF INPUT LINES:    {vald_nlin:6d}')
print(f'DATA INPUT: NUMBER OF SELECTED LINES: {len(wl):6d}')

#Retrieve levels used in the line list
my_e0 = np.array(list(set(ei) | set(ej))) # sorted level energies

#Selection only levels with energy lower than e_max    
my_e = my_e0[my_e0 < e_lim]

#Add the ionization level
my_e = list(my_e)
my_e.append(ion_el[elt])
my_e = sorted(my_e)

level = np.array(list(set(leveli) | set(levelj))) #Not used so far 



#print(f'\nNumber of unique lower levels: {len(set(leveli))}')
#print(f'Number of unique upper levels:   {len(set(levelj))}')
#print(f'Number of unique levels:         {len(level)}')

print(f'\nDATA INPUT:     NUMBER OF LEVELS: {len(my_e0)}')
print(f"DATA SELECTION: SELECT ONLY LEVELS WITH ENERGY LOWER THAN {e_max} {e_max_unit} ({e_lim:8.1f} cm⁻¹)")
print(f'DATA SELECTION: NUMBER OF LEVELS: {len(my_e)}')

#Initialize the model atom
fd = open(ofn, 'w')
print(f'{elt:10s} generated by formato3 {time.asctime()}', file=fd)
print(f'* {" ".join(arg for arg in sys.argv)}', file=fd)

print(f'* Energy level parameters: e_max = {e_max} [below 50: in eV; above 50: in cm⁻¹]', file=fd)
print(f'* Broadening parameters: t_broad = {t_broad} K, nh_broad = {nh_broad} /m^3, ne_broad = {ne_broad} /m^3', file=fd)
print(f'* Line frequency parameters: fcut = {fcut}, nq_min = {nq_min}, q0_min = {q0_min}, qmax_min = {qmax_min}', file=fd)
if elt in norad_px.keys():
    print(f'* Photoionization parameters: npi = {npi}, gaussian_width0 = {gaussian_width0}, std_frac = {std_frac}, tol_max = {tol_max}, resampling = {resampling}', file=fd)
print(f'* Collision parameters: t_coll  = {t_coll} K, sce_type = {sce_type}, sh = {sh}', file=fd)
if activate_loi and elt in loi.keys():
    print(f'* Lines of interest: {", ".join(map(str,loi[elt]))}', file=fd)

print('* ABUND   AWGT', file=fd)
print(f'  {abund[elt]:>5.2f}   {mass[elt]}', file=fd)
print('*  NK   NLIN  NCNT NFIX', file=fd)
print(f' {len(my_e):>4d} {len(set(wl)):>6d}  {len(my_e)-1:4d}    0', file=fd)

#Dictionary: key: level energies, value: number line in vald  
dic_ei = {}
for i, e in enumerate(ei):
    try:
        dic_ei[e].append(i)
    except KeyError:
        dic_ei[e] = [i]

dic_ej = {}
for j, e in enumerate(ej):
    try:
        dic_ej[e].append(j)
    except KeyError:
        dic_ej[e] = [j]

#dic_ei_ej = {} #key: (energy_level_i, energy_level_j) value: number line in vald
#for i, (eli, elj) in enumerate(zip(ei, ej)):
#    try:
#        dic_ei_ej[eli, elj].append(i)
#    except KeyError:
#        dic_ei_ej[eli, elj] = [i]


def select_term(terms):
    
    t = list(set(terms.copy()))

    #No term ambiguity
    if len(t) == 1:
        return t[0]
    else:
        #Some configuration in VALD are given with or without LS coupling for the same level
        #Select the one with LS term
        try:
            t.remove('')
        except ValueError:
            pass

        if len(t) == 1:
            return t[0]
        else:
            print(f'Conflicting terms for the same level: {t}')    
            quit()


print('*    E[cm-1]   G           CONFIGURATION & TERM              ION NK', file=fd)

#Energy levels
my_g, my_cfg, my_term = [], [], []
for nk, e in enumerate(my_e, start=0):
    i, j = -1, -1
    try:
        i = list(ei).index(e)
        mi = ei == e
    except ValueError:
        try:
            j = list(ej).index(e)
            mj = ej == e
        except ValueError: #Correspond to the ionization stage
            print(f"{e:12.4f} {ion_g[elt]:3d} '{ion_cfg[elt]:30s} {ion_term[elt]:10s}'  {int(elt[-1])+1}  {nk+1}", file=fd)
            print(f"DATA INPUT: IONIZATION LEVEL: {e:12.4f} {ion_g[elt]:3d} '{ion_cfg[elt]:30s} {ion_term[elt]:10s}' {int(elt[-1])+1} {nk+1}")
            my_g.append(ion_g[elt])
            my_cfg.append(ion_cfg[elt])
            my_term.append(ion_term[elt])
            nk_ion = nk #index of ionization level
            continue           

    if i >= 0:
        print(f"{e:12.4f} {gi[i]:3d} '{cfgi[i].strip():30s} {select_term(termi[mi]):10s}'  {elt[-1]}  {nk+1}", file=fd)
        my_g.append(gi[i])
        my_cfg.append(cfgi[i].strip())
        my_term.append(select_term(termi[mi]))    
    elif j>= 0:
        print(f"{e:12.4f} {gj[j]:3d} '{cfgj[j].strip():30s} {select_term(termj[mj]):10s}'  {elt[-1]}  {nk+1}", file=fd)
        my_g.append(gj[j])
        my_cfg.append(cfgj[j].strip())
        my_term.append(select_term(termj[mj])) 
    else:
        print(f'DATA INPUT: LEVEL ENERGY: {e} NOT FOUND IN ei OR ej')
        quit()

if write_el:
    for nk, (e, g, cfg, term) in enumerate(zip(my_e, my_g, my_cfg, my_term)):
        if e == ion_el[elt]:
            print(f"{e:12.4f} {g:3d} '{cfg.strip():30s} {term:10s}' {int(elt[-1])+1} {nk+1}")
        else:      
            print(f"{e:12.4f} {g:3d} '{cfg.strip():30s} {term:10s}' {elt[-1]} {nk+1}")


print(f'DATA INPUT: TERMS: {sorted(set(my_term))}')
#print(f'DATA INPUT: TERMSi: {sorted(set(termi))}')
#print(f'DATA INPUT: TERMSj: {sorted(set(termi))}')

#Functions for semi-classical collisions
def vac2air(x):
    ''' Conversion of wavelengths from vacuum to the air in Angström in the standard condition
        of pressure and temperature (IAU Morton (1991, ApJS, 77, 119))
        http://www.sdss.org/DR6/products/spectra/vacwavelength.html
    '''
    return x / (1.0 + 2.735182E-4 + 131.4182E0 / x**2 + 2.76249E8 / x**4)

def parent_cfg_wo_term(cfg):
    ''' Remove terms in parenthesis inside the configuration
         3d7.(4F).4p          => 3d7.4p
         3d6.(5D).4s.4p.(1P*) => 3d6.4s.4p
         3d6.4s.(6D<7/2>).4f  => 3d6.4s.4f
    '''
    pcfg = re.sub('\.[\(<][a-zA-Z0-9\*<>/]*[\)>\.|\)$]', '', cfg)
    return pcfg

def get_sqn(cfg):
    ''' Get the secondary quantum number from the electronic configuration
        get_sqn('3p6.4s2') returns 0
        get_sqn('3p6.4s.5p') returns 1
    '''
    sqn_list = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm']

    cfg = parent_cfg_wo_term(cfg)
    orbs = cfg.split('.')
    
    if orbs[-1] in ['1s2', '2s2', '3s2', '4s2', '5s2', '6s2', '7s2']:
        if len(orbs) == 1:
            orb = orbs[-1]
        else:
            if orbs[-2] in ['2p6', '3p6', '4p6', '5p6', '6p6', '7p6']:
                orb = orbs[-1]
            else:
                orb = orbs[-2]
    else:
        orb = orbs[-1]

    for c in orb:
        if c.isalpha() and c.islower():
            try:
                return sqn_list.index(c)
            except ValueError:
                pass

def voigt(x, w, dnud, gamma):
    #Voigt function, see https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/

    x0 =  cc/ w*1e-10 / dnud
    y = 1/(np.sqrt(np.pi)*dnud) * np.real(ss.wofz( x - x0 + 1j*gamma/(4*np.pi)/dnud)) 
    return y

def rad_broad(p, w):
    # p radiative damping parameter from VALD
    # w line wavelength
    
    if p == 0: #Empty parameter in VALD
        #Unsöld formula (1955) using Merle PhD report, eq. (3.79) formulation
        grad = 2.223e-5 / (w*1e-10)**2    
    elif p != 0:   
        #Unlog the VALD value
        grad = 10**p                           
   
    return grad

def col_broad_VdW(p, m, t, nh, ei, ej):
    # m mass of the non-LTE element
    # p Van der Waals damping parameter from VALD
    # t typical temperature in stellar atmospheres
    # nh typical hydrogen density in stellar atmospheres

    a0 = 5.2917720859e-11  # m

    if p < 0:
        #According to the definition given in VALD3 http://www.astro.uu.se/valdwiki/Vald3Format
        #gv = nh * 10**gv * t/10000
        gvdw = 10**p * (1e-12)**(2/5) # convert in SI units according to Merle, PhD report, p. 80
        gvdw = nh * gvdw * (t/10000)**(3/10) #/(4*np.pi) It will be divided by 4pi in frequencies routine
    elif p == 0:
        #Enhancement factor based on Edvardsson's prescription (1993)
        if elt[-1] == '1':
            ef = 1.4
        else:
            ef = 2.5
        #Unsöld formula (1955) using Merle PhD report, eqs. (3.103) and (3.106) formulation    
        C6 = 6.502e-46 * int(elt[-1])**2 * ry**2* (1/(ion_el[elt]-ej)**2 - 1/(ion_el[elt]-ei)**2) * 1e-4 # Putting ei and ej in units of m⁻¹
        C6 =abs(C6) #C6 should be negative, but this could appear in very specific cases when energy levels are higher than the ionization level
        gvdw = ef * 277.1 * C6**(2/5)* nh * t**(3/10) 
    else: 
        #ABO theory http://www.astro.uu.se/~barklem/howto.html
        sigma0 = int(p)     # xsection at 10 km/s
        alpha = p - sigma0  # index power law
        mu = mass['H 1'] * m * amu / (mass['H 1'] + m) # reduced mass
        vmean = np.sqrt(8*kb*t/(np.pi*mu))
        sigma0 = sigma0 * a0**2 # Convert from atomic units to SI
        gvdw = 2 * nh * (4/np.pi)**(alpha/2) * ss.gamma((4-alpha)/2) * 1e4 * sigma0 * (vmean/1e4)**(1-alpha)  
        #print('ABO', end=' ')

    return gvdw

def col_broad_Sta(p, t, ne, ei, ej):
    # p Stark damping parameter from VALD

    if p == 0:
        #Cowley (1971) formula using Merle PhD report, eq. (3.101) formulation
        gsta = 4.62e-10 * ne * int(elt[-1])**2 * ry**2* (1/(ion_el[elt]-ej)**2 - 1/(ion_el[elt]-ei)**2) * 1e-4 
    else:
       #According to the definition in VALD3 http://www.astro.uu.se/valdwiki/Vald3Format
        #gs = ne * 10**gs * t/10000
        gsta = 10**p * (1e-8)**(2/3) # convert in SI units 
        gsta = ne * gsta * (t/10000)**(1/6) #/(4*np.pi)  It will be divided by 4pi in frequencies routine

    return gsta

def frequencies(t, nh, ne, m, w, gr, gs, gv, ei, ej):
    #See Gray book fig. 11.4 to have order of magnitude of damping parameters
    #Broadening parameters gr, gs, and gv are the ones read from the VALD linelist

    #Doppler width (s⁻¹)
    dnud = np.sqrt(2*kb*t/m/amu) / (w*1e-10) 

    #print(gr, gs, gv)
    
    gr_ij = rad_broad(gr, w)                    #Radiative damping
    gv_ij = col_broad_VdW(gv, m, t, nh, ei, ej) #Collisional damping with H atoms
    gs_ij = col_broad_Sta(gs, t, ne, ei, ej)    #Collisional damping with free electrons

   #Total damping
    gt = gr_ij + gs_ij + gv_ij
    
    avoigt = gt / (4*np.pi) / dnud

    #From BPz and RE in get-broadening.f
    #if avoigt < 0.05:
    #    q0 = 2.0
    #    qmax = 2*q0
    #elif (avoigt >= 0.05) and (avoigt < 0.1): 
    #    q0 = 2.0
    #    qmax = 3*q0
    #elif (avoigt >= 0.1) and (avoigt < 1.0):
    #    q0 = 2.0 
    #    qmax = 5*q0 
    #elif (avoigt >= 1.0) and (avoigt < 10):
    #    q0 = 2.0   
    #    qmax = 10.0*q0
    #else:
    #    q0 = 2.0
    #    qmax = 25*q0

    x0 = cc / (dnud * w*1e10) # adimensional central wavelength/frequency
    dq = 0.5
    voigt_threshold = 0.1
    voigt_limit = 0.0001
    #q1 = np.arange(0, 20, dq)
    q1 = np.arange(0, 40, dq)
    voigt0 = voigt(x0, w, dnud, gt)
    voigt1 = voigt(x0 + q1, w, dnud, gt)
    m1 = voigt1/voigt0 > voigt_threshold
    try:
        q0, dq = max(q1[m1]), max(q1[m1])/5
    except ValueError:
        print(w, ei, ej, x0, gr_ij, gv_ij, gs_ij)
        quit()
    #q2 = np.arange(q0, 300, dq)
    q2 = np.arange(q0, 600, dq)
    voigt2 = voigt(x0 + q2, w, dnud, gt)
    m2 = voigt2/voigt0 > voigt_limit
    qmax = max(q2[m2])
    #nq = min(int(qmax/q0) + 5, 300)
    nq = min(int(qmax/q0) + 5, 600)

    if q0 == qmax:
        dq = qmax/(nq-1)
        print('q0 == qmax')
        q = np.arange(0, nq, qmax/(nq-1))
        voigtq = voigt(x0+q, w, dnud, gt)
    elif (qmax >= 0) and (q0 >= 0):
        half = 4
        dx = np.log10(10**(q0+0.5) * max(half, qmax-q0-half)) / (nq-1)
        q = []
        for i in range(1, nq+1):
            q.append((i-1)*dx + (10**((i-1)*dx) - 1)/(10**(q0+0.5)))   
        voigtq = voigt(x0 + q, w, dnud, gt)

    if write_bp:
        print(f'w = {w:12.4f}, dnud = {dnud:10.3e}, gr:{gr_ij:10.3e} + gs:{gs_ij:10.3e} + gv:{gv_ij:10.3e}   =  gt:{gt:10.3e} avoigt = {avoigt:10.3e}', end=' ')
        print(f'nq: {nq:4d} q0: {q0:4.1f} qmax: {qmax:5.1f}')

    #Plot line frequencies for lines of interest   
    if wloi:
        
        if (abs(w - wloi) < 5e-3).any():
            
            #print(f'w = {w:12.4f}, dnud = {dnud:10.3e}, gr:{gr_ij:10.3e} + gs:{gs_ij:10.3e} + gv:{gv_ij:10.3e}   =  gt:{gt:10.3e} avoigt = {avoigt:10.3e}', end=' ')
            #print(f'nq: {nq:4d} q0: {q0:4.1f} qmax: {qmax:5.1f}')
            
            import matplotlib.pyplot as plt
            plt.rc('font', size=17)
            with plt.style.context('dark_background'):
                plt.figure(1, dpi=200, tight_layout=True)
                plt.plot(q1[m1], voigt1[m1]/voigt0, '.c')
                plt.plot(q2[m2], voigt2[m2]/voigt0, '.r')
                plt.plot(q, voigtq/voigt0, 'ow', alpha=1,zorder=1)
                plt.axhline(y=voigt_threshold, color='c', alpha=1, ls='dotted')
                plt.axhline(y=voigt_limit, color='r', alpha=1, ls='dotted')
                plt.axvline(x=q0, color='w', ls='dotted')
                plt.xlabel('$q$')
                plt.ylabel('H/H$_0$')
                plt.title(f'$\\lambda$ = {w} \u00C5, nq = {nq}, q0 = {q0:4.2f}, qmax = {qmax:4.2f}', fontsize=14)
                plt.savefig(f'voigt_{w}.png')
                plt.yscale('log') 
                plt.savefig(f'voigt_{w}_log.png')
                if plot_freq:
                    plt.show()
                else:
                    plt.close()

    return nq, q0, qmax



#Radiative bb transitions
print('* RADIATIVE B-B TRANSITIONS', file=fd)
print('*  J    I          F    NQ  QMAX    Q0 IW         GA       GVW         GS     LAMBDA[Å]     KR    LBD(VALD)  UP LOW   LOGGF REF', file=fd)
print('\nRADIATIVE BB TRANSITIONS')
kr = 0 #Number of radiative transition
nd = 0 #Number of duplicates 

try:
    wloi = loi[elt]
except KeyError:
    wloi = None

idx_lines_duplicate = []
nq_list, qmax_list, q0_list = [], [], []
for eli, idx_lines in sorted(dic_ei.items(), key=lambda t: t[0]): #Sort of the dictionnary per key (energy levels)
    ndpl = 0 #Number of duplicate per level
    ni = list(my_e).index(eli) + 1
    idx_lines_done = [] #Allow to check if a new line has a duplicate
    idx_lines_duplicate_per_level = []
    for i in idx_lines[::-1]:
        k = list(set(dic_ei[eli]) & set(dic_ej[ej[i]])) # list all the lines rising from two given level energies
        if i in idx_lines_done: # Duplicate with lower loggf values
            idx_lines_duplicate.append(i)
            idx_lines_duplicate_per_level.append(i)
            nd += 1
            ndpl += 1 #Number of dulplicates per level
            continue 
        else:
            idx_lines_done.extend(k)
            k = k[np.argmax(loggf[k])] #Avoid multiple line from the same two levels (take the one with the highest loggf), now k is the index in the vald data    
            kr += 1
        nj = list(my_e).index(ej[k]) + 1
        #Oscillator strength
        f = 10**(loggf[k])/gi[k]
        if f <  fcut:
            nq, q0, qmax = nq_min, q0_min, qmax_min
        else:
            nq, q0, qmax = frequencies(t_broad, nh_broad, ne_broad, mass[elt], wl[k], gr[k], gs[k], gv[k], ei[k], ej[k])
            nq_list.append(nq)
            qmax_list.append(qmax)
            q0_list.append(q0)
        #Compute wavelength from energy levels
        if my_e[ni-1] == 0:
            wl_from_el = vac2air((1/my_e[nj-1])*1e8)
        else:
            wl_from_el = vac2air((1/(my_e[nj-1] - my_e[ni-1])*1e8))
        #Secondary quantum number
        print(my_cfg[ni-1], my_cfg[nj-1])
        sqn_i = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm'][get_sqn(my_cfg[ni-1])]
        sqn_j = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm'][get_sqn(my_cfg[nj-1])]
        #print(wl[k], wl_from_el, my_cfg[ni-1], sqn_i, my_cfg[nj-1], sqn_j)
        
        #Broadening parameters to write in the model atom with format for MULTI
        
        gr_ij = rad_broad(gr[k], wl[k]) #Radiative broadening

        #Standard enhancement factor for Van der Waals broadening based on Edvardsson's prescription (1993)
        if (gv[k] >= 0.) and (gv[k] < 20):
            if elt[-1] == '1':
                gv_ij = 1.4
            else:
                gv_ij = 2.5 
        else:
            #Use the parameter from VALD directly because transformation already encoded in MULTI 2.3
            gv_ij = gv[k]
        
        #Stark broadening
        if gs[k] == 0.:
            #Use the parameter from VALD directly because transformation already encoded in MULTI 2.3
            gs_ij = gs[k]
        else:
            gs_ij = col_broad_Sta(gs[k], t_broad, ne_broad, ei[k], ej[k])/(ne_broad/1e6) # ne_broad converted in /cm^3 to be consistent with units in MULTI

        #If line of astrophyiscal interest exist for that ion, set the q0 = qmax to force frequency linear scale and nq to nq_loi
        iw = 0 # for normal bound-bound transition
        if activate_loi:
            if wloi:
                if (abs(wl[k] - wloi) < 5e-3).any():   
                    q0 = qmax
                    nq = nq_loi
                    iw = 1 # for transitions with wavelength dependend background opacities 

        print(f"{nj:4d} {ni:4d} {f:10.3e} {nq:5d} {qmax:5.1f} {q0:5.2f} {iw:2d} {gr_ij:10.3e} {gv_ij:9.3f} {gs_ij:10.3e}  {wl_from_el:12.4f} {kr:6d} {wl[k]:12.4f} '{sqn_j}' '{sqn_i}' {loggf_ref[k]}", file=fd)
    #for i in idx_lines_duplicate_per_level:
    #    print(valdline[i])
    #    print(valdline[i+1])
    print(f'LEVEL {ni:3d}: {len(idx_lines)-ndpl:5d} TRANSITIONS TOWARD THIS LEVEL (NUMBER OF REMOVED DUPLICATE: {len(idx_lines_duplicate_per_level):3d})')


print(f'{kr:6d} RADIATIVE TRANSITIONS')
print(f'{nd:6d} DUPLICATE TRANSITIONS')
print(f'RADIATIVE TRANSITIONS: MIN(NQ)   = {min(nq_list):4d},     MAX(NQ) = {max(nq_list):4d}, MEAN(NQ) =    {np.mean(nq_list):4.1f}')
print(f'RADIATIVE TRANSITIONS: MIN(Q0)   =  {min(q0_list):4.1f},    MAX(Q0) = {max(q0_list):4.1f}, MEAN(Q0) =    {np.mean(q0_list):4.1f}')
print(f'RADIATIVE TRANSITIONS: MIN(QMAX) = {min(qmax_list):5.1f}, MAX(QMAX) = {max(qmax_list):5.1f}, MEAN(QMAX) = {np.mean(qmax_list):5.1f}')

#quit()


#Function for semi-classical photoionization
def gl_bf(l, m_minus, eta, rho):
    '''from Janicki et al. 1991'''
    b = np.zeros(210)
    gl = 0
    
    if l == 0:
        return gl
    
    m = -m_minus

    b[1] = 1.
    b[2] = 2*m*eta / l

    for i in np.arange(2, 2*m+1):
        b[i+1] = -(4*eta*(i-1-m)*b[i] + (2*m+2-i)*(2*m+2*l+1-i)*b[i-1]) / (i*(i+2*l-1))

    for i in np.arange(0, 2*m+1):
        gl += b[i+1]*rho**i

    return gl

def gaunt_bf(n, l, Z, Eion):
    '''Gaunt factor from Janicki et al. 1991
    '''

    #Tweak when the principal quantum number is not known: n = 0
    #to avoid error on negative factorial   
    if n == 0:
        n = l + 1

    eta = np.sqrt(Z**2 * 13.6057 / Eion)
    rho = eta / n
    res = 0
    for i in np.arange(1, l+1):
        res += np.log10(i**2 + eta**2)

    r12 = 1 + rho**2
    el2 = eta**2 + (l+1)**2

    #print(n, l)
    
    facln1 = np.log(float(np.math.factorial(n + l))) #the float() method is needed to avoid overflow (TypeError: loop of ufunc does not support argument 0 of type int which has no callable log method)
    facln2 = np.log(np.math.factorial(2*l + 1))
    facln3 = np.log(np.math.factorial(2*l + 2))
    facln4 = np.log(float(np.math.factorial(n - l - 1)))

    a1 = np.exp(-2*eta*(np.pi/2 - np.math.atan(rho)) + 0.5*(facln1 - facln2 - facln3 -facln4 + res + l*np.log(16)) + (l-1)*np.log(rho) - (n-2)*np.log(r12))
    a2 = (0.3401*n/el2) / (1 - np.exp(-2*np.pi*eta))
    a3 = 4*l**3 *(l+1) * (2*l+1) / (l**2+eta**2)
    a4 = 64 * (l+1)**2 * (rho*eta/r12)**2 / ((2*l+1)*el2)

    gl1 = gl_bf(l, l+1-n, eta, rho)
    gl2 = gl_bf(l, l-1-n, eta, rho)
    gl3 = gl_bf(l+1, l+1-n, eta, rho)
    gl4 = gl_bf(l+1, l-n, eta, rho)

    #print(f'gl: {gl1} {gl2} {gl3} {gl4}')

    gbf = a2*(a3*(a1*gl1-a1*gl2/r12**2)**2 + a4*((l+1-n)*a1*gl3 + a1*gl4*(l+1+n)/r12)**2)

    return gbf

def gaunt_menzel_and_pickeris(Z, n, Eion):
    ''' Gaunt factor from Menzel & Pickeris (1935)
        n: principal quantum number
        Eion: Ionization energy [cm⁻¹]
        Z: effective ionization degree
    '''
    neff = Z * np.sqrt(ry/(100*abs(Eion)))
    #print('neff:', neff)
    return 1.0 - 0.1728*(1/neff)**0.333 * (2*(neff/n)**2-1)

def gaunt_karzas_and_latter(Z, n, l, Eion):
    ''' Gaunt factor from Karzas & Latter (1961)
        n: principal quantum number
        l: secundary quantum number
        Eion: Ionization energy [cm⁻¹]
        Z: effective ionization degree
    '''
    Eion = 100*hp*cc/qe * Eion # eV
    #print('Eion:', Eion)
    return gaunt_bf(n, l, Z, Eion)

def sigma0_bf_kramers(n, Eion, gbf):
    ''' Semi-classical relation from Kramers 
        (modified version of Travis & Matsushima (1968) eq. 18 and 19)
        Return photoionisation cross-section at threshold in m²
        Eion: ionization energy of the level [cm⁻¹]
        Z: ionic charge
        n: principal quantum number
        gbf: bound-free Gaunt factor
    '''

    #Tweak when the principal quantum number is not known: n = 0
    #to avoid error on negative factorial   
    if n == 0:
        n = l + 1

    nu0 = Eion * 100 * cc 
    #return 2.815e25 *  Z**4 /(n**5 * nu0**3)
    return 2.815e25 * (Eion*100/ry)**2 *1/(n*nu0**3) * gbf



def get_pqn(cfg):
    ''' Get the principal quantum number from the electronic configuration
        get_pqn('3p6.4s2') returns 4
        get_pqn('3p6.4s.10p') returns 10
        get_pqn('3s2.3p.(2P*<1/2>).6s.<1/2>') returns
    '''
    cfg = parent_cfg_wo_term(cfg)
    orbs = cfg.split('.')
    norb = len(orbs)
    orb = orbs[norb-1]
    m = re.search('^[0-9n]*', orb)
    if m:
        try:
            return int(m.group())
        except ValueError: #case where pqn is n
            return 0

def idx_kramer(e, x):
    for i, sign in enumerate(np.sign(np.gradient(x, e))[::-1]):
        if sign > -1.:
            i_kramer = x.size - i
            break
    return i_kramer

def smooth(x, i1, i2, width):
    sx = x.copy()
    sx[i1:i2] = sn.gaussian_filter1d(x[i1:i2], width)
    sx[sx != sx] = x[sx != sx]
    return sx

def os(e, x, i1, i2, thres):
    ose, osx = [], []
    ose.extend(e[:i_min])
    osx.extend(x[:i_min])
    ce_store, cx_store = [], []
    for ce, cx in zip(e[i1:i2], x[i1:i2]):
        ce_store.append(ce)
        cx_store.append(cx)
        if len(cx_store) > 1 and (len(cx_store)%2 == 1):
            if np.std(cx_store) > thres or (np.max(ce_store) - np.min(ce_store) > 0.05):
                med = np.median(cx_store)
                i = cx_store.index(med)
                ose.append(ce_store[i])
                osx.append(cx_store[i])         
                ce_store = []
                cx_store = []
    ose.extend(e[i2:])
    osx.extend(x[i2:])
    ose = np.array(ose)
    osx = np.array(osx)
    return ose, osx

def use_os_px(e0, x0,i_min, npi,std_frac, gaussian_width0, resampling):
    #Dictionnary of the input energies
    dic_e0 = {}
    for j, e in enumerate(e0):
        try:
            dic_e0[e].append(j)
        except KeyError:
            dic_e0[e] = [j]
    
    #Cleaning of the input data
    e1, x1 = [], []
    for e, j  in dic_e0.items():
        if len(j) == 1:
            x = x0[j[0]]
        elif len(j) > 1:
            x = np.mean(x0[j])
        else:
            print(e, j, sum(j))
            quit(1)
        e1.append(e)
        x1.append(x)
    e1, x1 = np.array(e1), np.array(x1)
    #print(f'Number of original input data:  {e0.size}')
    #print(f'Number of cleaned input data:   {e1.size}')
    int_x0 = np.trapz(x0, e0)
    int_x1 = np.trapz(x1, e1)
    #print(f'original integral:         {int_x0:10.4e} {x0.size:4d}')
    #print(f'original cleaned integral: {int_x1:10.4e} {x1.size:4d}')
    #print(f'Input parameters: i_min: {i_min} npi: {npi} std_frac: {std_frac} gaussian_width0: {gaussian_width0} resampling: {resampling}')
    
    #Index for the Kramer's decreasing law
    i_kramer = idx_kramer(e1, x1)
    
    if resampling:
        #Resampling original clean data
        re = np.linspace(e1[:i_kramer].min(), e1[:i_kramer].max(), e1[:i_kramer].size)
        rx = si.interp1d(e1[:i_kramer], x1[:i_kramer])(re)
        re = list(re)
        rx = list(rx)
        re.extend(e1[i_kramer:])
        rx.extend(x1[i_kramer:])
        re = np.array(re)
        rx = np.array(rx)
        e1 = re.copy()
        x1 = rx.copy()
        i_kramer = idx_kramer(e1, x1)
    #Loop over the number of points and tolerance
    ose, osx = e1.copy(), x1.copy()
    i = 0
    gaussian_width = gaussian_width0
    while osx.size > npi or tol > tol_max:
        i += 1
        #Smoothing of cleaned original data
        sx = smooth(x1, i_min, i_kramer, gaussian_width)
        #Opacity sampling
        disp_max = std_frac*np.std(sx)*i
        ose, osx = os(e1, sx, i_min, i_kramer, disp_max)
        #Opacity sampling integral
        int_osx = np.trapz(osx, ose)
        tol = abs(int_x1-int_osx)/int_x1
        
        print(f'{i:3d} {int_osx:10.4e} {tol*100:5.2f}% {tol_max:4.2} {osx.size:4d} {gaussian_width:3.0f} {disp_max:5.2e}')
    
        gaussian_width += 2

        #To avoid infinite loop
        if gaussian_width >= gaussian_width0 + 40:
            break

    return ose, osx


print('* RADIATIVE B-F TRANSITIONS', file=fd)
print('* ION   I    A0   NPTS', file=fd)
print('\nRADIATIVE BF TRANSITIONS')

#Photoionization with Gaunt factor formula and Kramer's law for default value
my_sbf = []
for nk, (e, g, cfg, term) in enumerate(zip(my_e, my_g, my_cfg, my_term), start=1):
    
    if e == ion_el[elt]: #Avoid photoionization of the ionization level
        my_sbf.append(0.)
        continue

    n = get_pqn(cfg)
    l = get_sqn(cfg)
    Z = int(elt[-1])
    Eion = ion_el[elt] - e
    #print(e, g, cfg, term)
    #print(Z, n, l, Eion)
    gaunt_kl = gaunt_karzas_and_latter(Z, n, l, abs(Eion))      #Absolute value for level above ionization threshold
    sigma0_kl = sigma0_bf_kramers(n, abs(Eion), gaunt_kl)*1e22  #Absolute value for level above ionization threshold
    sbf = sigma0_kl*1e-18
    my_sbf.append(sbf)

#Read norad energy levels and photoionization if exist
try:
    norad_px[elt]
    with open(norad_en[elt], 'r') as fd1:
        content1 = fd1.read().splitlines()
    fd1.close()
    with open(norad_px[elt], 'r') as fd2:
        content2 = fd2.read().splitlines()
    fd2.close()

    norad_flag_ion = True
    norad_multiplicity, norad_angular_momentum, norad_parity, norad_nel, norad_iel, norad_el = [], [], [], [], [], []
    
    #Read the energy
    for i, record in enumerate(content1):
        if record == '    0    0    0    0': #End of file
            break
        if norad_flag_ion: #Read negative ionization enery in Rydberg of the ground state 
            match = re.findall(r'Eo\(Ry\) = *([0-9E+-.]+) *=-IP', record)
            if match != []:
                norad_ion_e_ryd = float(match[0])
                norad_flag_ion = False
    
        match = re.match(r'^ *([0-9]) *([0-9]) *([01]) *([0-9]+) *$', record) # Read Norad spectral term 
        if match:
            idata = match.groups()
            for j in np.arange(2, int(idata[3])+2): # Read all the energy level in a given spectral term
                norad_iel.append(int(content1[i+j].split()[0]))
                norad_el.append(float(content1[i+j].split()[5])) # Ionization energy
                norad_multiplicity.append(int(idata[0]))
                norad_angular_momentum.append(int(idata[1]))
                norad_parity.append(int(idata[2]))
                norad_nel.append(int(idata[3]))
                #print(f'{norad_multiplicity[-1]} {norad_angular_momentum[-1]} {norad_parity[-1]} {norad_nel[-1]:2d} {norad_iel[-1]:2d} {norad_el[-1]:12.5e}')
    norad_multiplicity = np.array(norad_multiplicity)
    norad_angular_momentum = np.array(norad_angular_momentum)
    norad_parity = np.array(norad_parity)
    norad_nel = np.array(norad_nel)
    norad_iel = np.array(norad_iel)
    norad_el  = np.array(norad_el)

    norad_ion_el_cm = abs(norad_el * ry / 100) 

    #Read and store header of photoionization xsections
    px_header, px_idx = [], []
    for j, record in enumerate(content2):
        match = re.match(f'(^ *[0-9] *[0-9] *[01] *[0-9]+ *$)', record)
        if match:
            px_header.append(match.groups()[0])
            px_idx.append(j)
    #print(px_header, px_idx)
except KeyError:
    pass

#Photoionization with quantum data
el_px_done = []
for nk, (e, g, cfg, term) in enumerate(zip(my_e, my_g, my_cfg, my_term), start=1):
    
    if e == ion_el[elt]: #Avoid photoionization of the ionization level
        continue

    vald_ion_el_cm = ion_el[elt]-e #Ionization energy of the given level
    #print(f'{nk} {e:12.4f} {g:3d} {cfg:30s} {term:10s} {vald_ion_el_cm:12.4f}')
    print(f'PHOTOIONIZATION OF LEVEL {nk:4d}: {e:12.4f} {g:3d} {cfg:30s} {term:10s} {vald_ion_el_cm:12.4f}', end=' ')

    m = []
    try:
        norad_px[elt]
        try:
            multiplicity = int(term[0])
        except ValueError:
            multiplicity = -1
        try:
            angular_momentum = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', '['].index(term[1])
        except IndexError:
            angular_momentum = -1
        try:
            parity = [0, 1].index(len(term)-2)
        except ValueError:
            parity = -1

        m = norad_multiplicity == multiplicity 
        m &= norad_angular_momentum == angular_momentum
        m &= norad_parity == parity

        if (sum(m) > 0) and (g <= norad_multiplicity[m][0]*(2*norad_angular_momentum[m][0]+1)):
            de_cm = norad_ion_el_cm[m] - vald_ion_el_cm
            i  = abs(de_cm).argmin()
            mul = norad_multiplicity[m][i]
            am = norad_angular_momentum[m][i]
            par = norad_parity[m][i]
            nel = norad_nel[m][i]
            iel = norad_iel[m][i]
            el = norad_el[m][i]
            el_cm = norad_ion_el_cm[m][i] 
            print(f'NORAD: {mul} {am} {par} {nel:2d} {iel:2d} {el:12.5f} {el_cm:12.4f} {de_cm[i]:12.4f}')
            el_px_done.append(el)
            #Read photoionization table
            for idx, header in zip(px_idx, px_header):
                match = re.match(f'^ *{mul} *{am} *{par} *{iel} *$', header)
                if match:
                    #print(match)
                    ntot = int(content2[idx+1].split()[1])
                    e0, x0 = [], []                    
                    for k in np.arange(2, ntot+3):
                        e0.append(float(content2[idx+k].split()[0]))
                        x0.append(float(content2[idx+k].split()[1]))
                    #print(e0[-1],x0[-1])
                    e0 = np.array(e0)
                    x0 = np.array(x0)
            
            ose, osx = use_os_px(e0, x0,i_min, npi,std_frac, gaussian_width0, resampling) # Ry, Mbarn

            ose = 1e10 / (ose * ry) # in Å
            osx = 1e-18 * osx #  in cm²

            print(f'{list(my_e).index(ion_el[elt]) + 1:4d} {nk:3d} {osx[0]:8.3e} {ose.size:4d} -1', file=fd)
            for j in range(ose.size):
                print(f'{ose[j]:8.2f} {osx[j]:8.3e}', file=fd)

        else:
            #print(f"{list(my_e).index(ion_el[elt]) + 1:4d} {nk:3d} {sbf:8.3e} 11 911.0")
            print(f"GAUNT FACTOR + KRAMER'S LAW: {my_sbf[nk-1]:8.3e} cm2")
            print(f"{list(my_e).index(ion_el[elt]) + 1:4d} {nk:3d} {my_sbf[nk-1]:8.3e} 11 911.0", file=fd)
    except KeyError:
        #print(f"{list(my_e).index(ion_el[elt]) + 1:4d} {nk:3d} {sbf:8.3e} 11 911.0")
        print(f"GAUNT FACTOR + KRAMER'S LAW: {my_sbf[nk-1]:8.3e} cm2")        
        #print(f"{list(my_e).index(ion_el[elt]) + 1:4d} {nk:3d} {my_sbf[nk-1]:8.3e} 11 911.0, {my_cfg[nk-1]}, {cfg} {term} {my_term[nk-1]}")       
        print(f"{list(my_e).index(ion_el[elt]) + 1:4d} {nk:3d} {my_sbf[nk-1]:8.3e} 11 911.0", file=fd)       

try:
    norad_px[elt]
    print(f'NUMBER OF QUANTUM PHOTOIONIZATION TABLES USED: {len(set(el_px_done))}/{len(norad_el)}')
    print(f'NUMBER OF LEVELS WITH QUANTUM PHOTOIONIZATION: {len(el_px_done)}')
except KeyError:
    pass



def neff(e_ion, e_level, Z):
    ''' Compute effective principal quantum number
        e_ion: ionization energy in cm⁻¹
        e_level: level energy cm⁻¹
        Z: ionization degree of the atom
    '''
    e = 100 * hp * cc / qe * abs(e_ion - e_level) # Ionization energy of the level in eV
    return Z * np.sqrt(13.6057/e) 

def orbital_radius(e_ion, e_level, Z, cfg_level):
    ''' Normally only valid for hydrogenoid ions
        e_ion: ionization energy in cm⁻¹
        e_level: level energy cm⁻¹
    '''
    ns = neff(e_ion, e_level, Z)
    l = get_sqn(cfg_level)
    return (3*ns**2-l*(l+1)) / (2*Z) # in unit of Cst.A0 (Bohr radius)

def phi(x):
    ''' Function for the IPM (Seaton, 1962)
    '''
    return x * ss.k0(x) * ss.k1(x)

def zeta(x):
    ''' Function for the IPM (Seaton, 1962)
    '''
    return x**2 * (ss.k0(x)**2 + ss.k1(x)**2)

def beta0(x, x0, r, T):
    ''' argument of phi function for weak coupling
        x: energy of colliding electron before collision [without dimension]
        x0: energy of the collision transition [without dimension]
        r: hydrogenoid orbital radius
        T: temperature [K]
    '''
    #return r * pl.sqrt(Cst.K * T * (x+x0)/13.6057/Cst.Q) * x0 / (2*x + x0)
    return r * np.sqrt(kb * T * (x)/13.6057/qe) * x0 / (2*x - x0)

def func_seaton(b1, c1):
    '''Implicit equation to solve for Seaton IPM method
    '''
    return ss.k0(b1)**2 + ss.k1(b1)**2 - c1

def beta1(x, x0, f, T):
    ''' argument of the implicit equation for the strong coupling
        x: energy of colliding electron before collision [without dimension]
        x0: energy of the collision transition [without dimension]
        f: oscillator strength
        T: temperature [K]
    '''
    #c1 = Cst.K * T * x0 / (8. * Cst.RYD * Cst.H * Cst.C * f) * ((2*x+x0)/x0)**2
    c1 = kb * T * x0 / (8. * ry * hp * cc * f) * ((2*x-x0)/x0)**2
    b1 = so.brentq(func_seaton, 0, 1000, args=c1)
    return b1

def sigma_weak_seaton(x, e_ion, eli, cfg_eli, elj, cfg_elj, w, f, T):
    ''' Compute collision cross-section for weak coupling in the IPM formula (Seaton 1962)
        x: energy of colliding electron before collision [wo dimension]
        e_ion: ionization energy in cm⁻¹
        eli: low level energy cm⁻¹
        cfg_eli: electronic configuration of the lower level
        elj: low level energy cm⁻¹
        cfg_elj: electronic configuration of the upper level
        w: wavelength [Å]
        f: oscillator strenght [wo dimension]
        T: temperature [K]
    '''
    cst = hp * cc / (qe * 1e-10)
    
    r_i = orbital_radius(e_ion, eli, Z, cfg_eli)
    r_j = orbital_radius(e_ion, elj, Z, cfg_elj)
    #print(r_i, r_j)
    
    if r_i <= r_j:
        r = r_i
    else:
        r = r_j   
    
    de = cst / w
    x0  = de * qe / kb / T
    #print(r, de, x0)
    b0 = beta0(x, x0, r, T)
    if x < x0:
        return 0.
    else:
        #return 8 * (13.6057*Cst.Q/Cst.K/T)**2 * line.f * (1/x0) * (1/(x+x0)) * phi(b0)
        return 8 * (13.6057*qe/kb/T)**2 * f * (1/x0) * (1/(x)) * phi(b0)

def sigma_strong_seaton(x, w, f, T):
    ''' Compute collision cross-section for strong coupling in the IPM formula (Seaton 1962)
        x: energy of colliding electron before collision [without dimension]
        w: wavelength [Å]
        f: oscillator strenght [wo dimension]
        T: temperature [K]
    '''
    cst = hp * cc / (qe * 1e-10)
    de = cst / w
    x0  = de * qe / kb / T    
    b1 = beta1(x, x0, f, T)
    if x < x0:
        return 0.
    else:
        #return 8 * (13.6057*Cst.Q/Cst.K/T)**2 * line.f * (1/x0) * (1/(x+x0)) * (0.5*zeta(b1) + phi(b1))
        return 8 * (13.6057*qe/kb/T)**2 * f * (1/x0) * (1/(x)) * (0.5*zeta(b1) + phi(b1))

def sigma_seaton(x, e_ion, eli, cfg_eli, elj, cfg_elj, w, f, T):
    ''' Select the lowest cross-section from the weak and strong coupling
        x: energy of colliding electron before collision [without dimension]
        e_ion: ionization energy in cm⁻¹
        eli: low level energy cm⁻¹
        cfg_eli: electronic configuration of the lower level
        elj: low level energy cm⁻¹
        cfg_elj: electronic configuration of the upper level
        w: wavelength [Å]
        f: oscillator strenght [wo dimension]
        T: temperature [K]
    '''
    q0 = sigma_weak_seaton(x, e_ion, eli, cfg_eli, elj, cfg_elj, w, f, T)
    q1 = sigma_strong_seaton(x, w, f, T)

    if q0 < q1:
        return q0
    else:
        return q1

def ups_sce_seaton(e_ion, eli, cfg_eli, elj, cfg_elj, w, g_eli, f):
    ''' Integrate from the IPM cross-sections
        e_ion: ionization energy in cm⁻¹
        eli: low level energy cm⁻¹
        cfg_eli: electronic configuration of the lower level
        elj: low level energy cm⁻¹
        cfg_elj: electronic configuration of the upper level
        w: wavelength [Å]
        g_eli: statistical weight of the lower level
        f: oscillator strenght [wo dimension]
    '''
    cst = hp * cc / (qe * 1e-10)
    de = cst / w
    def func(T):
        x0  = de * qe / kb / T
        #x_vec = pl.linspace(0.0001, 100, 300)
        x_vec_log = np.linspace(-5, 4, 100)
        x_vec = 10**x_vec_log
        q_vec = [sigma_seaton(i+x0, e_ion, eli, cfg_eli, elj, cfg_elj, w, f, T) for i in x_vec]
        ups_cst = g_eli * kb * T / ry / hp / cc 
        return ups_cst * np.trapz(q_vec*(x_vec+x0)*np.exp(-(x_vec)), x_vec)
        #return ups_cst * pl.trapz(q_vec*(x_vec)*pl.exp(-(x_vec-x0)), x_vec)
    return func

def ups_sce_vanregemorter(gi, f, Eo, deg):
    ''' Compute the effectice electron collision strength
        with semi-classical formula from Van Regemorter 1962, ApJ
        Eo [eV] transition energy
    '''    
    cst = 197.42/Eo * gi * f
    def func(T):
        #return cst * average_gaunt_vanregemorter(deg, Eo, T)
        return cst * average_gaunt_vanregemorter2(deg, Eo, T)
    return func

def average_gaunt_vanregemorter2(deg, Eo, T):
    ''' Compute the maxwellian average of the Gaunt factor for electron collision
    '''
    x_vec = np.linspace(0.2, 8., 50) # sqrt(E/Eo)
    x0 = Eo * qe / (kb * T)    # Eo/kT
    g_func = make_gaunt_vanregemorter(deg)
    g_vec = [g_func(x) * np.exp(-x**2*x0) for x in x_vec]

    return np.trapz(g_vec, x_vec**2*x0)

def make_gaunt_vanregemorter(deg):
    ''' according to Van Regemorter 1962, ApJ, Table 1. and Fig. 1
    '''
    x = [0.0, 0.2, 0.4, 0.6, 0.8, 1., 2., 3., 4., 5., 6., 7., 8.]# sqrt(E/Eo)
    #x = [0.04, 0.16, 0.36, 0.64, 1., 4., 9., 16., 25., 36.] # E/Eo
    gi = [0.0, 0.015, 0.034, 0.057, 0.084, 0.124, 0.328, 0.561, 0.775, 0.922, 1.040, 1.12, 1.20]
    gii = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.328, 0.561, 0.775, 0.922, 1.040, 1.12, 1.20]

    if deg == '1':
        return si.interp1d(x, gi, kind='cubic')
    elif deg == '2':
        return si.interp1d(x, gii, kind='cubic')
    else:
        print("Z must be '1' or '2'")
        quit(1)

#Collisions
print(f'* COLLISION TRANSITIONS:', file=fd)
print(f'\nCOLLISION TRANSITIONS: TEMPERATURE RANGE: {t_coll}')
print(f'GENCOL', file=fd)

print(f'* ELECTRON BB  COLLISIONS: {sce_type}', file=fd)
print(f'ELECTRON BB COLISIONS: {sce_type}')

print('TEMP', file=fd)
print(f'{len(t_coll)} {" ".join([str(t) for t in t_coll])}', file=fd)

#dic_i_j = {}
#for i in range(len(my_e)):
#    for j in np.arange(i+1, len(my_e)):
#        try:
#            k = list(set(dic_ei[my_e[i]]) & set(dic_ej[my_e[j]]))
#        except KeyError:
#            k = []
#        dic_i_j[i, j] = k
#
#print('ca commence')
#na, nf = 0, 0
#for (i, j), k in sorted(dic_i_j.items(), key=lambda t: t[0]):
#    #print(i, j, k)
#    if i == 11:
#        break
#    if k == []:
#        nf += 1
#        ups = list(np.ones(len(t_coll)))
#        w = vac2air(1e8/(my_e[j] -my_e[i]))    
#        print(f'UPS_E      DEFAULT  {w:12.4f} FORBIDDEN', file=fd) 
#    else:
#        na += 1
#        k = k[np.argmax(loggf[k])] #Avoid multiple line from the same two levels (take the one with the highest loggf)
#        #k = np.argmax(m) 
#        upse = ups_sce_seaton(ion_el[elt], my_e[i], my_cfg[i], my_e[j], my_cfg[j], wl[k], gi[i], 10**(loggf[k])/gi[i]) 
#        ups = [upse(float(t)) for t in t_coll]
#        print(f'UPS_E     SEATON   {wl[k]:12.4f} ALLOWED', file=fd)
#    print(f'{j+1:4d} {i+1:4d} {" ".join([format(u, "10.3e") for u in ups])}', file=fd)        
#print("c'est fini !")

na, nf = 0, 0 # Number allowed, number forbidden
for i, eli in enumerate(my_e):  
    napl, nfpl = 0, 0 # Number allowed per level, number forbidden per level
    for j, elj in enumerate(my_e[i+1:], start=i+1):
        #m = (ei == eli) & (ej == elj)
        if elj == ion_el[elt]: #Avoid electronic BB collision with the ionization level
            continue
        try:
            k = list(set(dic_ei[eli]) & set(dic_ej[elj]))
            #k = dic_ei_ej[eli, elj]
        except KeyError:
            k = []
        #if sum(m) == 0:
        if k == []:
            nf += 1
            nfpl += 1
            ups = list(np.ones(len(t_coll)))
            w = vac2air(1e8/(elj - eli))    
            print(     f'UPS_E      DEFAULT         {w:12.4f} FORBIDDEN', file=fd) 
        #elif sum(m) == 1:
        else:
            na += 1
            napl += 1
            k = k[np.argmax(loggf[k])] #Avoid multiple line from the same two levels (take the one with the highest loggf)
            #k = np.argmax(m) 
            f = 10**(loggf[k])/my_g[i]
            if sce_type == 'SEATON':
                upse = ups_sce_seaton(ion_el[elt], eli, my_cfg[i], elj, my_cfg[j], wl[k], my_g[i], f) 
                print(f'UPS_E      SEATON           {wl[k]:12.4f} ALLOWED', file=fd)
            elif sce_type == 'VAN REGEMORTER':
                Eo = hp * cc / qe * (my_e[j] - my_e[i]) *100
                upse = ups_sce_vanregemorter(my_g[i], f, Eo, elt[-1])
                print(f'UPS_E      VAN REGEMORTER   {wl[k]:12.4f} ALLOWED', file=fd)
            else:
                print(f'sce_type = {sce_type} unknown')
                quit(1)
            ups = [upse(float(t)) for t in t_coll]

        #else:
        #    print(f'sum(m) = {sum(m)}')
        #    continue
        print(f'{j+1:4d} {i+1:4d} {" ".join([format(u, "10.3e") for u in ups])}', file=fd)
    print(f'LEVEL {i+1:4d}: {eli:12.4f} {my_g[i]:5d} {my_cfg[i]:30s} {my_term[i]:10s}: {napl:6d} ALLOWED TRANSITIONS, {nfpl:6d} FORBIDDEN TRANSITIONS, {napl+nfpl:6d} TOTAL TRANSITIONS')
        
print(f'TOTAL: {na:6d} ALLOWED TRANSITIONS, {nf:6d} FORBIDDEN TRANSITIONS, {na+nf} TOTAL TRANSITIONS.')

print('* ELECTRON BF  COLLISIONS', file=fd)
print('ELECTRON BF  COLLISIONS')

for ni, eli in enumerate(my_e):
    # For the moment just copy the bf cross-section at the threshold in unit of cm⁻²
    # To improve 
    # Based on Mihalas 78 itself based on Seaton in Bates 62
    # Also found in Mashonkina 1996

    if eli == ion_el[elt]: #Avoid BF collision of the ionization level with itself
        continue

    w = abs(vac2air(1e8/(ion_el[elt] - eli)))
    print(f'CI         SEATON   {w:12.4f} from PHOTO XSECTION')    
    print(f'CI         SEATON   {w:12.4f} from PHOTO XSECTION', file=fd)    
    print(f'{ni+1:4d} {list(my_e).index(ion_el[elt]) + 1:4d} {my_sbf[ni]:10.3e}', file=fd)



def svm_sch_drawin(gi, gj, f, Eo, mass):
    '''Downward avarege bb collisions rates
    '''
    mh = 1.008*amu
    ma = mass*amu

    cst = 4*np.sqrt(2)* me * ma / mh**2 * ry * hp * cc / qe * 2.014e-7

    def func(T):
        '''Downward rates
        '''
        x = Eo/(kb * T/ qe)
        return cst * gi/gj * f /np.sqrt(T) / Eo * (1+2/x) / x
    return func

def svm_schi_drawin(gi, gj, Eo, mass):
    '''Downward average bf collisions rates
       Three body recombination with hydrogen 
    '''
    mh = 1.008*amu
    ma = mass*amu

    #bf oscillator strength:
    f = 1
    #Number of equivalent electron in the outer shell to be ionized
    xi = 1

    cst = 4*np.sqrt(2)* me * ma / mh**2 * ry * hp * cc / qe * 2.014e-7

    def func(T):
        '''Downward rates
        '''
        x = Eo/(kb * T/ qe)
        return cst * gi/gj * f * xi /np.sqrt(T) / Eo * (1+2/x) / x
    return func


if sh:

    print('* HYDROGEN BB  COLLISIONS', file=fd)
    print('\nHYDROGEN BB  COLLISIONS')
    
    kr = 0 #Number of allowed collision transition
    nd = 0 #Number of duplicates 
    for eli, idx_lines in sorted(dic_ei.items(), key=lambda t: t[0]): #Sort of the dictionnary per key (energy levels)
        ndpl = 0
        ni = list(my_e).index(eli) # index of low level
        idx_lines_done = [] #Allow to check if a new line as a duplicate
        for i in idx_lines[::-1]:
            #Find the right k index for the allowed transition
            k = list(set(dic_ei[eli]) & set(dic_ej[ej[i]])) # list all the lines rising from two given level energies
            if i in idx_lines_done: # Duplicate with lower loggf values
                nd += 1
                ndpl += 1 #Number of dulplicates per level
                continue 
            else:
                idx_lines_done.extend(k)
                k = k[np.argmax(loggf[k])] #Avoid multiple line from the same two levels (take the one with the highest loggf), now k is the index in the vald data    
                kr += 1
            #Compute the Sigma V Mean svm   
            nj = list(my_e).index(ej[k])
            Eo = hp * cc / qe / (wl[k] * 1e-10)
            f = 10**loggf[k]/gi[k]
            svmh = svm_sch_drawin(gi[k], gj[k], f, Eo, mass[elt])
            svm = [sh*svmh(float(t)) for t in t_coll]
            print(f"CH         DRAWIN   {wl[k]:12.4f} ALLOWED", file=fd)
            print(f'{nj+1:4d} {ni+1:4d} {" ".join([format(s, "10.3e") for s in svm])}', file=fd)
        #print(f'level {ni:3d}: {len(idx_lines)-ndpl:5d} transitions toward this level.')
        print(f'LEVEL {ni+1:4d}: {eli:12.4f} {my_g[ni]:5d} {my_cfg[ni]:30s} {my_term[ni]:10s}: {kr:6d} ALLOWED TRANSITIONS')


    print('* HYDROGEN BF  COLLISIONS', file=fd)
    print('HYDROGEN BF  COLLISIONS')

    for ni, eli in enumerate(my_e):
        if eli == ion_el[elt]: #Avoid BF collision of the ionization level with itself
            continue   
        w = abs(vac2air(1e8/(ion_el[elt] - eli)))
        Eo = hp * cc / qe * (my_e[nk_ion] - my_e[ni]) *100 #Ionization energy of the level in eV
        svmh = svm_schi_drawin(my_g[ni], my_g[nk_ion], Eo, mass[elt])
        svm = [sh*svmh(float(t)) for t in t_coll]
        print(f'CHI        DRAWIN   {w:12.4f} IONIZATION WITH F=1 and XI=1')    
        print(f'CHI        DRAWIN   {w:12.4f} IONIZATION WITH F=1 and XI=1', file=fd)    
        print(f'{ni+1:4d} {list(my_e).index(ion_el[elt]) + 1:4d} {" ".join([format(s, "10.3e") for s in svm])}', file=fd)       

print('END', file=fd)



fd.close()

print(f'ATOM: {elt}')

if len(set(wl)) != kr:
    print(f'WARNING: NLIN IN THE MODEL ATOM MUST BE CHANGED FROM {len(set(wl))} INTO {kr}')