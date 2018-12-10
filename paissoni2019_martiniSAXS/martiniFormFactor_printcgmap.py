#!/usr/bin/env python


###################################
## 1 # OPTIONS AND DOCUMENTATION ##  -> @DOC <-
###################################
import sys,logging
logging.basicConfig(format='%(levelname)-7s    %(message)s',level=9)
from scipy.optimize import curve_fit
import os.path

    
# This is a simple and versatily option class that allows easy
# definition and parsing of options. 
class Option:
    def __init__(self,func=str,num=1,default=None,description=""):
        self.func        = func
        self.num         = num
        self.value       = default
        self.description = description
    def __nonzero__(self): 
        if self.func == bool:
            return self.value != False
        return bool(self.value)
    def __str__(self):
        return self.value and str(self.value) or ""
    def setvalue(self,v):
        if len(v) == 1:
            self.value = self.func(v[0])
        else:
            self.value = [ self.func(i) for i in v ]
    
options = [
    ("-filelist", Option(str,   1,     None, "Input file list, if this option is present all the other options are useless")),
    ("-f",        Option(str,   1,     None, "Input file (Must be a GRO file)")),
    ("-x",        Option(str,   1,     "cg.pdb", "Output coarse grained structure (PDB)")),
    ("-ff",       Option(str,   1,     "ff.out", "Output Form Factor")),
    ("-xind",     Option(str,   1,     "cgIndex.out", "Output coarse grained index")),
    ("-ffind",    Option(str,   1,     "ffIndex.out", "Output Form Factor index")),
    ("-xexc",     Option(str,   1,     "cgExcl.out", "Output coarse grained excluded")),
    ("-ffexc",    Option(str,   1,     "ffExcl.out", "Output Form Factor excluded")),  
    ]


def option_parser(args,options):

    # Convert the option list to a dictionary, discarding all comments
    options = dict([i for i in options if not type(i) == str])
    options['Arguments']           = args[:]
    while args:
        ar = args.pop(0)
        options[ar].setvalue([args.pop(0) for i in range(options[ar].num)])
    
    return options 


def option_parser2(f,fname,options):
    # Convert the option list to a dictionary, discarding all comments
    myargs = [ '-f', str(f), '-x', str("cg_{}.pdb".format(fname)), '-ff', str("ff_{}.out".format(fname)), 
            '-xind', str("cgInd_{}.out".format(fname)), '-ffind', str("ffInd_{}.out".format(fname)), 
            '-xexc', str("cgExcl_{}.out".format(fname)), '-ffexc', str("ffExcl_{}.out".format(fname)), '-fname', str("{}".format(fname)) ]
    options = dict([i for i in options if not type(i) == str])
    options['-fname'] =  Option(str,1,None)
    options['Arguments']           = myargs[:]
    while myargs:
        ar = myargs.pop(0)
        options[ar].setvalue([myargs.pop(0) for i in range(options[ar].num)])
    return options 



#################################################
## 2 # HELPER FUNCTIONS, CLASSES AND SHORTCUTS ##  -> @FUNC <-
#################################################

import math

# Split a string                                                              
def spl(x):                                                                   
    return x.split()                                                          

# Split each argument in a list                                               
def nsplit(*x):                                                               
    return [i.split() for i in x]                                             

# Make a dictionary from two lists                                            
def hash(x,y):                                                                
    return dict(zip(x,y))                                                     

# Function to reformat pattern strings                                        
def pat(x,c="."):                                                             
    return x.replace(c,"\x00").split()                                        

# Function to generate formatted strings according to the argument type       
def formatString(i):                                                          
    if type(i) == str:                                                        
        return i                                                              
    if type(i) == int:                                                        
        return "%5d"%i                                                        
    if type(i) == float and 0<abs(i)<1e-5:                                                      
        return "%2.1e"%i                                                      
    elif type(i) == float:                                                      
        return "%8.5f"%i                                                      
    else:                                                                     
        return str(i)                                                         



def concatFiles(fwrite,flist,exclude):
    #with open(fwrite,'w') as outfile:
    for fn in flist:
        with open(fn,'r') as infile:
            for line in infile:
                if exclude not in line:
                    fwrite.write(line)



#----+----------------+
## B | MATH FUNCTIONS |
#----+----------------+

def norm2(a):
    return sum([i*i for i in a])


def norm(a):
    return math.sqrt(norm2(a))


def distance2(a,b):
    return (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2


##########################
## 3 # FG -> CG MAPPING ##  -> @MAP <-
##########################


dnares3 = " DA DC DG DT" 
dnares1 = " dA dC dG dT"
rnares3 = "  A  C  G  U"
rnares1 = " rA rC rG rU" # 

# Amino acid nucleic acid codes:                                                                                 
# The naming (AA and '3') is not strictly correct when adding DNA/RNA, but we keep it like this for consistincy.
AA3     = spl("TRP TYR PHE HIS HIH ARG LYS CYS ASP GLU ILE LEU MET ASN PRO HYP GLN SER THR VAL ALA GLY"+dnares3+rnares3) #@#
AA1     = spl("  W   Y   F   H   H   R   K   C   D   E   I   L   M   N   P   O   Q   S   T   V   A   G"+dnares1+rnares1) #@#


# Dictionaries for conversion from one letter code to three letter code v.v.                         
AA123, AA321 = hash(AA1,AA3),hash(AA3,AA1)                                                           


# Residue classes:
protein = AA3[:-8]   # remove eight to get rid of DNA/RNA here.
water   = spl("HOH SOL TIP")
lipids  = spl("DPP DHP DLP DMP DSP POP DOP DAP DUP DPP DHP DLP DMP DSP PPC DSM DSD DSS")
nucleic = spl("DAD DCY DGU DTH ADE CYT GUA THY URA DA DC DG DT A C G U")


residueTypes = dict(
    [(i,"Protein") for i in protein ]+
    [(i,"Water")   for i in water   ]+
    [(i,"Lipid")   for i in lipids  ]+
    [(i,"Nucleic") for i in nucleic ]
    )

class CoarseGrained:
    # Class for mapping an atomistic residue list to a coarsegrained one
    # Should get an __init__ function taking a residuelist, atomlist, Pymol selection or ChemPy model
    # The result should be stored in a list-type attribute
    # The class should have pdbstr and grostr methods

    # Standard mapping groups
    bb        = "N CA C O H H1 H2 H3 O1 O2 OC1 OC2"                                                                    #@#  
    dna_bb = "P OP1 OP2 O5' O3'","C5' O4' C4'","C3' C2' C1'"
    rna_bb = "P OP1 OP2 O5' O3'","C5' O4' C4'","C3' C2' O2' C1'"


    # This is the mapping dictionary
    # For each residue it returns a list, each element of which
    # lists the atom names to be mapped to the corresponding bead.
    # The order should be the standard order of the coarse grained
    # beads for the residue. Only atom names matching with those 
    # present in the list of atoms for the residue will be used
    # to determine the bead position. This adds flexibility to the
    # approach, as a single definition can be used for different 
    # states of a residue (e.g., GLU/GLUH).
    # For convenience, the list can be specified as a set of strings,
    # converted into a list of lists by 'nsplit' defined above.
    mapping = {
        "ALA":  nsplit(bb + " CB"),
        "CYS":  nsplit(bb,"CB SG"),
        "ASP":  nsplit(bb,"CB CG OD1 OD2"),
        "GLU":  nsplit(bb,"CB CG CD OE1 OE2"),
        "PHE":  nsplit(bb,"CB CG CD1 HD1","CD2 HD2 CE2 HE2","CE1 HE1 CZ HZ"),
        "GLY":  nsplit(bb),
        "HIS":  nsplit(bb,"CB CG","CD2 HD2 NE2 HE2","ND1 HD1 CE1 HE1"),
        "HIH":  nsplit(bb,"CB CG","CD2 HD2 NE2 HE2","ND1 HD1 CE1 HE1"),     # Charged Histidine.
        "ILE":  nsplit(bb,"CB CG1 CG2 CD CD1"),
        "LYS":  nsplit(bb,"CB CG CD","CE NZ HZ1 HZ2 HZ3"),
        "LEU":  nsplit(bb,"CB CG CD1 CD2"),
        "MET":  nsplit(bb,"CB CG SD CE"),
        "ASN":  nsplit(bb,"CB CG ND1 ND2 OD1 OD2 HD11 HD12 HD21 HD22"),
        "PRO":  nsplit(bb,"CB CG CD"),
        "HYP":  nsplit(bb,"CB CG CD OD"),
        "GLN":  nsplit(bb,"CB CG CD OE1 OE2 NE1 NE2 HE11 HE12 HE21 HE22"),
        "ARG":  nsplit(bb,"CB CG CD","NE HE CZ NH1 NH2 HH11 HH12 HH21 HH22"),    
        "SER":  nsplit(bb,"CB OG HG"),
        "THR":  nsplit(bb,"CB OG1 HG1 CG2"),
        "VAL":  nsplit(bb,"CB CG1 CG2"),
        "TRP":  nsplit(bb,"CB CG CD2","CD1 HD1 NE1 HE1 CE2","CE3 HE3 CZ3 HZ3","CZ2 HZ2 CH2 HH2"),
        "TYR":  nsplit(bb,"CB CG CD1 HD1","CD2 HD2 CE2 HE2","CE1 HE1 CZ OH HH"),
        "DA": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' C1'",
                          "N9 C4",
                          "C2 N3",
                          "C6 N6 N1",
                          "C8 N7 C5"),
        "DG": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' C1'",
                          "N9 C4",
                          "C2 N2 N3",
                          "C6 O6 N1",
                          "C8 N7 C5"),
        "DC": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' C1'",
                          "N1 C6",
                          "N3 C2 O2",
                          "C5 C4 N4"),
        "DT": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' C1'",
                          "N1 C6",
                          "N3 C2 O2",
                          "C5 C4 O4 C7 C5M"),
         "A":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' O2' C1'",
                          "N9 C4",
                          "C2 N3",
                          "C6 N6 N1",
                          "C8 N7 C5"),
        "G":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' O2' C1'",
                          "N9 C4",
                          "C2 N2 N3",
                          "C6 O6 N1",
                          "C8 N7 C5"),
        "C":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' O2' C1'",
                          "N1 C6",
                          "N3 C2 O2",
                          "C5 C4 N4"),
        "U":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4'",
                          "C3' C2' O2' C1'",
                          "N1 C6",
                          "N3 C2 O2",
                          "C5 C4 O4 C7 C5M"),
        }



    # This is the mapping dictionary including Hydrogen atoms!
    bbHA        = "N CA C O H H1 H2 H3 O1 O2 HA HA1 HA2 OC1 OC2"                    
    mappingH = {
        "ALA":  nsplit(bbHA + " CB HB1 HB2 HB3"),
        "CYS":  nsplit(bbHA,"CB SG HB1 HB2 HG"),
        "ASP":  nsplit(bbHA,"CB CG OD1 OD2 HB1 HB2"),
        "GLU":  nsplit(bbHA,"CB CG CD OE1 OE2 HB1 HB2 HG1 HG2"),
        "PHE":  nsplit(bbHA,"CB CG CD1 HD1 HB1 HB2","CD2 HD2 CE2 HE2","CE1 HE1 CZ HZ"),
        "GLY":  nsplit(bbHA),
        "HIS":  nsplit(bbHA,"CB  HB1 HB2 CG","CD2 HD2 NE2 HE2","ND1 HD1 CE1 HE1"),
        #"HIH":  nsplit(bbHA,"CB CG","CD2 HD2 NE2 HE2","ND1 HD1 CE1 HE1"),     # Charged Histidine.
        "ILE":  nsplit(bbHA,"CB CG1 CG2 CD CD1 HB HG21 HG22 HG23 HG11 HG12 HG13 HD1 HD2 HD3"),
        "LYS":  nsplit(bbHA,"CB CG CD HB1 HB2 HG1 HG2 HD1 HD2","CE HE1 HE2 NZ HZ1 HZ2 HZ3"),
        "LEU":  nsplit(bbHA,"CB CG CD1 CD2 HB1 HB2 HG HD11 HD12 HD13 HD21 HD22 HD23"),
        "MET":  nsplit(bbHA,"CB CG SD CE HB1 HB2 HG1 HG2 HE1 HE2 HE3"),
        "ASN":  nsplit(bbHA,"CB HB1 HB2 CG ND1 ND2 OD1 OD2 HD11 HD12 HD21 HD22"),
        "PRO":  nsplit(bbHA,"CB CG CD HB1 HB2 HG1 HG2 HD1 HD2"),
        "HYP":  nsplit(bbHA,"CB CG CD OD HB1 HB2 HG1 HG2 HD1 HD2 HD"),
        "GLN":  nsplit(bbHA,"CB CG HB1 HB2 HG1 HG2 CD OE1 OE2 NE1 NE2 HE11 HE12 HE21 HE22"),        
        "ARG":  nsplit(bbHA,"CB CG CD HB1 HB2 HG1 HG2 HD1 HD2","NE HE CZ NH1 NH2 HH11 HH12 HH21 HH22"),    
        "SER":  nsplit(bbHA,"CB HB1 HB2 OG HG"),
        "THR":  nsplit(bbHA,"CB HB OG1 HG1 CG2 HG21 HG22 HG23 OG2 HG2 CG1 HG11 HG12 HG13"),
        "VAL":  nsplit(bbHA,"CB CG1 CG2 HB HG21 HG22 HG23 HG11 HG12 HG13"),
        "TRP":  nsplit(bbHA,"CB CG CD2 HB1 HB2","CD1 HD1 NE1 HE1 CE2","CE3 HE3 CZ3 HZ3","CZ2 HZ2 CH2 HH2"),
        "TYR":  nsplit(bbHA,"CB HB1 HB2 CG CD1 HD1","CD2 HD2 CE2 HE2","CE1 HE1 CZ OH HH"),
        # elimino H3T altrimenti me lo associa a O3' sbagliato! dovro' trovare una soluzione in futuro
        "DA": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' C1' H1' H3' H2'1 H2'2",
                          "N9 C4",
                          "C2 N3 H2",
                          "C6 N6 N1 H61 H62",
                          "C8 N7 C5 H8"),
        "DG": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' C1' H1' H3' H2'1 H2'2",
                          "N9 C4",
                          "C2 N2 N3 H21 H22",
                          "C6 O6 N1 H1",
                          "C8 N7 C5 H8"),
        "DC": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' C1' H1' H3' H2'1 H2'2",
                          "N1 C6 H6",
                          "N3 C2 O2",
                          "C5 C4 N4 H5 H41 H42"),
        "DT": nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' C1' H1' H3' H2'1 H2'2",
                          "N1 C6 H6",
                          "N3 C2 O2 H3",
                          "C5 C4 O4 C7 C5M H71 H72 H73 H5"),
        "A":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' O2' C1' H1' H3' H2' H2'1 HO'2",
                          "N9 C4",
                          "C2 N3 H2",
                          "C6 N6 N1 H61 H62",
                          "C8 N7 C5 H8"),
        "G":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' O2' C1' H1' H3' H2' H2'1 HO'2",
                          "N9 C4",
                          "C2 N2 N3 H21 H22",
                          "C6 O6 N1 H1",
                          "C8 N7 C5 H8"),
        "C":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' O2' C1' H1' H3' H2' H2'1 HO'2",
                          "N1 C6 H6",
                          "N3 C2 O2",
                          "C5 C4 N4 H5 H41 H42"),
        "U":  nsplit("P OP1 OP2 O5' O3' O1P O2P",
                          "C5' O4' C4' H5'1 H5'2 H4'",
                          "C3' C2' O2' C1' H1' H3' H2' H2'1 HO'2",
                          "N1 C6 H6",
                          "N3 C2 O2 H3",
                          "C5 C4 O4 C7 C5M H71 H72 H73 H5"),
        } 


    # Generic names for side chain beads
    residue_bead_names = spl("BB SC1 SC2 SC3 SC4")
    # Generic names for DNA/RNA beads
    residue_bead_names_dna = spl("BB1 BB2 BB3 SC1 SC2 SC3 SC4")
    residue_bead_names_rna = spl("BB1 BB2 BB3 SC1 SC2 SC3 SC4")

    # This dictionary contains the bead names for all residues,
    # following the order in 'mapping'
    # Add default bead names for all amino acids
    names = {}
    names.update([(i,("BB","SC1","SC2","SC3","SC4")) for i in AA3])
    # Add the default bead names for all DNA nucleic acids
    names.update([(i,("BB1","BB2","BB3","SC1","SC2","SC3","SC4")) for i in nucleic])

    # Crude mass for weighted average. No consideration of united atoms.
    # This will probably give only minor deviations, while also giving less headache
    mass = {'H': 1,'C': 12,'N': 14,'O': 16,'S': 32,'P': 31,'M': 0}
    

class AtomicFormFactor:

    # ATOMIC FORM FACTOR
    # REFERENCES
    ##### theory/formula etc.:
    # Computational and Structural Biotechnology Journal,  Vol 8, Issue 11, e201308006, http://dx.doi.org/10.5936/csbj.201308006. Reconstruction of SAXS Profiles from Protein Structures
    # Fraser et al. J. Appl. Cryst. (1978). 11, 693-694 
    ##### a/b/c parameters (ATT! there are two options for H)
    # International Tables for Crystalloraphy (2006). Vol. C, Chapter 6.1, pp. 554-595 
    ##### Volume v
    # Fraser et al. J. Appl. Cryst. (1978). 11, 693-694 
    # J. Appl. Cryst. (1995), 28, 768-733. CRYSOL 
    ##### Mean electron density of bulk water [rho]
    # Computational and Structural Biotechnology Journal,  Vol 8, Issue 11, e201308006, http://dx.doi.org/10.5936/csbj.201308006. Reconstruction of SAXS Profiles from Protein Structures
    # see also: https://books.google.it/books?id=gTAWBAAAQBAJ&pg=PA593&lpg=PA593&dq=water+mean+electron+density+0.334&source=bl&ots=x_E9rMwEcY&sig=bpsvc3oh7-ZWlzGleW5NaI9lWh0&hl=it&sa=X&ved=0ahUKEwjdx6WRhr_aAhWkw6YKHQzUAVoQ6AEIMDAB#v=onepage&q=water%20mean%20electron%20density%200.334&f=false
    # rho=0.334 e-/Ang3 for pure water; rho=0.40  e-/Ang3 for salt solution with higher density; crystallographic programs often use a compromise value around 0.375 e-/Ang3
    # Here we use 0.334; consider the possibility to use 0.375

    param_a = []; param_b = []
    for i in range(5):
        param_a.append({ })
        param_b.append({ })
    param_c = { }
    param_v = { }

    rho=0.334

    # H parameters to reproduce spherical bonded H-atoms scattering factors from Stewart, Davidson and Simpson (1965)
    # Table 6.1.1.2 and 6.1.1.3 of International Tables for Crystalloraphy (2006). Vol. C, Chapter 6.1, pp. 554-595
    param_a[0]['H'] = 0.493002; param_b[0]['H'] = 10.5109; param_c['H'] = 0.003038;
    param_a[1]['H'] = 0.322912; param_b[1]['H'] = 26.1257; param_v['H'] = 5.15;
    param_a[2]['H'] = 0.140191; param_b[2]['H'] = 3.14236;
    param_a[3]['H'] = 0.040810; param_b[3]['H'] = 57.7997;
    
    # H parameters to reproduce H scattering factor calculated from the analytical solution to the Schr equation
    # It does not consider correction to scattering factor for bonded H (for other atoms this correction is less relevant)
    # Table 6.1.1.1 and 6.1.1.3 of International Tables for Crystalloraphy (2006). Vol. C, Chapter 6.1, pp. 554-595
    # RESULTS ARE ALMOST IDENTICAL USING THIS OR CORRECTED SCATTERING FACTORS
    #param_a[0]['H'] = 0.489918; param_b[0]['H'] = 20.6593; param_c['H'] = 0.001305;
    #param_a[1]['H'] = 0.262003; param_b[1]['H'] = 7.74039; param_v['H'] = 5.15;
    #param_a[2]['H'] = 0.196767; param_b[2]['H'] = 49.5519;
    #param_a[3]['H'] = 0.049879; param_b[3]['H'] = 2.20159

    param_a[0]['C'] = 2.31000; param_b[0]['C'] = 20.8439; param_c['C'] = 0.215600;
    param_a[1]['C'] = 1.02000; param_b[1]['C'] = 10.2075; param_v['C'] = 16.44;
    param_a[2]['C'] = 1.58860; param_b[2]['C'] = 0.56870;
    param_a[3]['C'] = 0.86500; param_b[3]['C'] = 51.6512;

    param_a[0]['N'] = 12.2126; param_b[0]['N'] = 0.00570; param_c['N'] = -11.529;
    param_a[1]['N'] = 3.13220; param_b[1]['N'] = 9.89330; param_v['N'] = 2.49;
    param_a[2]['N'] = 2.01250; param_b[2]['N'] = 28.9975;
    param_a[3]['N'] = 1.16630; param_b[3]['N'] = 0.58260;

    param_a[0]['O'] = 3.04850; param_b[0]['O'] = 13.2771; param_c['O'] = 0.250800 ;
    param_a[1]['O'] = 2.28680; param_b[1]['O'] = 5.70110; param_v['O'] = 9.13;
    param_a[2]['O'] = 1.54630; param_b[2]['O'] = 0.32390;
    param_a[3]['O'] = 0.86700; param_b[3]['O'] = 32.9089;

    param_a[0]['P'] = 6.43450; param_b[0]['P'] = 1.90670; param_c['P'] = 1.11490;
    param_a[1]['P'] = 4.17910; param_b[1]['P'] = 27.1570; param_v['P'] = 5.73;
    param_a[2]['P'] = 1.78000; param_b[2]['P'] = 0.52600;
    param_a[3]['P'] = 1.49080; param_b[3]['P'] = 68.1645;

    param_a[0]['S'] = 6.90530; param_b[0]['S'] = 1.46790; param_c['S'] = 0.866900;
    param_a[1]['S'] = 5.20340; param_b[1]['S'] = 22.2151; param_v['S'] = 19.86;
    param_a[2]['S'] = 1.43790; param_b[2]['S'] = 0.25360;
    param_a[3]['S'] = 1.58630; param_b[3]['S'] = 56.1720;


def FormFactor(atomn,qlist):
    f = np.ones(len(qlist))
    f = np.multiply(f,AtomicFormFactor.param_c[atomn])
    s = np.power(qlist/(4.*math.pi),2.)
    volt = np.power(AtomicFormFactor.param_v[atomn],2./3.)/(4.*math.pi)
    for k in range(4):
        f += np.multiply(AtomicFormFactor.param_a[k][atomn],np.exp(np.multiply(-AtomicFormFactor.param_b[k][atomn],s)))
    f -= np.multiply(AtomicFormFactor.rho*AtomicFormFactor.param_v[atomn],np.exp(np.multiply(-volt,np.power(qlist,2.))))
    return f
   

# Determine average position for a set of weights and coordinates
# This is a rather specific function that requires a list of items
# [(m,(x,y,z),id),..] and returns the weighted average of the 
# coordinates and the list of ids mapped to this bead
def aver(b):
    mwx,ids,atom,aids = zip(*[((m*x,m*y,m*z),i,at,aid) for m,(x,y,z),i,at,aid in b])              # Weighted coordinates     
    tm  = sum(zip(*b)[0])                                                 # Sum of weights           
    return [sum(i)/tm for i in zip(*mwx)],ids,atom,aids                            # Centre of mass           

# function for sixth-order polinomial fit
def func6(x,a0,a1,a2,a3,a4,a5,a6):
    return a0+a1*x+a2*x**2+a3*x**3+a4*x**4+a5*x**5+a6*x**6

# Determine the bead form factors
def beadff(b,qval,atomicff):
    beadffv=0; beadff0=0;
    for a1 in b:
            ids,atom,aids = zip(*[(i,at,aid) for t,(x,y,z),i,at,aid in b])
            beadff0 +=  atomicff[a1[0]][0]
            for a2 in b:
                if a2[4] >= a1[4]:
                    qr=np.multiply(qval,np.sqrt(distance2(a1[1],a2[1])))
                    qr_over_pi=np.divide(qr,np.pi) # needed to use np.sinc
                    if a2[4] == a1[4]:
                        beadffv +=  np.multiply(np.multiply(atomicff[a1[0]],atomicff[a2[0]]),np.sinc(qr_over_pi))
                    else:
                        beadffv +=  2*np.multiply(np.multiply(atomicff[a1[0]],atomicff[a2[0]]),np.sinc(qr_over_pi))
    beadffsq =np.sqrt(beadffv)
    #do fit if beadff0<0
    if np.abs(beadff0-beadffsq[0]) > np.finfo(np.float32).eps:
        if beadff0 >= 0:
            logging.error("Something is wrong in the form factor calculations for q=0. Exiting...")
        else:
            ind=np.diff(beadffsq).argmax()
            onev=np.ones(7);
            lowb=np.multiply(onev,-np.inf); lowb[0]=beadff0;
            upb=np.multiply(onev,np.inf); upb[0]=beadff0+np.finfo(np.float32).eps;
            popt, pcov = curve_fit(func6, qval[ind+1:], beadffsq[ind+1:], bounds=(lowb,upb))
            beadffsq=func6(qval, *popt)
    return beadffsq,ids,atom,aids                            

# Return the CG beads for an atomistic residue, using the mapping specified above
# The residue 'r' is simply a list of atoms, and each atom is a list:
# [ name, resname, resid, chain, x, y, z ]
def mapCG(r):
    p = CoarseGrained.mapping[r[0][1]]     # Mapping for this residue 
    flat_p = [item for sublist in p for item in sublist]
    # Get the atom_name, mass, coordinates (x,y,z), atom id for all atoms in the residue
    a = [(i[0],CoarseGrained.mass.get(i[0][0],0),i[4:7],i[7]) for i in r]               
    # Store weight, coordinate and index for atoms that match a bead
    q = [[(m,coord,a.index((atom,m,coord,aid)),atom,aid) for atom,m,coord,aid in a if atom in i] for i in p]
    # Store atomName, atomID, resName, resID, chain and coord for atoms that DO NOT match a bead
    NOTinb = [(atom,aid) for atom,m,coord,aid in a if atom not in flat_p]
    # Return 
    # pos: bead positions, index, atom name, atom id  for atoms in beads 
    # NOTinb: atom name, atom id for atom NOT in beads
    return {'pos': zip(*[aver(i) for i in q if len(i)>0]), 'NOTinb': NOTinb }

def mapFF(r,qval,atomicff):
    p = CoarseGrained.mappingH[r[0][1]]     # Mapping for this residue
    flat_p = [item for sublist in p for item in sublist]
    # Get the atom_name, type (H,N,C,O,P,S), coordinates (x,y,z), atom id for all atoms in the residue
    a = [(i[0],i[0][0],i[4:7],i[7]) for i in r]  
    # Store type, coordinate and index for atoms that match a bead
    q = [[(t,coord,a.index((atom,t,coord,aid)),atom,aid) for atom,t,coord,aid in a if atom in i] for i in p]
    # Store atomName, atomID, resName, resID, chain and coord for atoms that DO NOT match a bead
    NOTinb = [(atom,aid) for atom,m,coord,aid in a if atom not in flat_p]
    # Return 
    # pos: bead positions, index, atom name, atom id  for atoms in beads 
    # NOTinb: atom name, atom id for atom NOT in beads 
    return {'ff': zip(*[beadff(i,qval,atomicff) for i in q if len(i)>0]) , 'NOTinb': NOTinb }


# Mapping for index file
def mapIndex(r):
    #print r[0][1]
    p = CoarseGrained.mapping[r[0][1]]                                             # Mapping for this residue 
    a = [(i[0],CoarseGrained.mass.get(i[0][0],0),i[4:7]) for i in r]                    
    # Store weight, coordinate and index for atoms that match a bead
    return [[(m,coord,a.index((atom,m,coord))) for atom,m,coord in a if atom in i] for i in p]

# Mapping for index file
def mapIndexH(r):
    #print r[0][1]
    p = CoarseGrained.mappingH[r[0][1]]                                             # Mapping for this residue 
    a = [(i[0],CoarseGrained.mass.get(i[0][0],0),i[4:7]) for i in r]                    
    # Store weight, coordinate and index for atoms that match a bead
    return [[(m,coord,a.index((atom,m,coord))) for atom,m,coord in a if atom in i] for i in p]



#######################
## 4 # STRUCTURE I/O ##  -> @IO <-
#######################
import logging,math,random,sys

#----+---------+
## A | PDB I/O |
#----+---------+

# Reformatting of lines in structure file                                     
pdbAtomLine = "ATOM  %5d %4s%4s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f\n"        

def pdbAtom(a):
    ##01234567890123456789012345678901234567890123456789012345678901234567890123456789
    ##ATOM   2155 HH11 ARG C 203     116.140  48.800   6.280  1.00  0.00
    if a.startswith("TER"):
        return 0
    # NOTE: The 27th field of an ATOM line in the PDB definition can contain an
    #       insertion code. We shift that 20 bits and add it to the residue number
    #       to ensure that residue numbers will be unique.
    ## ===> atom name,       res name,        res id,                        chain,
    return (a[12:16].strip(),a[17:20].strip(),int(a[22:26])+(ord(a[26])<<20),a[21],
    ##            x,              y,              z       
    float(a[30:38]),float(a[38:46]),float(a[46:54]),int(a[7:12]))

# Function for splitting a PDB file in chains, based
# on chain identifiers and TER statements
def pdbChains(pdbAtomList):
    chain = []
    for atom in pdbAtomList:
        if not atom: # Was a "TER" statement
            if chain:
                yield chain
            else:
                logging.info("Skipping empty chain definition")
            chain = [] 
            continue
        if not chain or chain[-1][3] == atom[3]:
            chain.append(atom)
        else:
            yield chain
            chain = [atom]
    if chain:
        yield chain


# Simple PDB iterator
def pdbFrameIterator(streamIterator):  
    title, atoms = [], []
    for i in streamIterator:
        if i.startswith("ENDMDL"):
            yield "".join(title), atoms
            title, atoms = [], []         
        elif i.startswith("TITLE"):
            title.append(i)
        elif i.startswith("ATOM") or i.startswith("HETATM"):
            atoms.append(pdbAtom(i))
    if atoms:
        yield "".join(title), atoms


#----+---------+
## B | GRO I/O |
#----+---------+

groline = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"                                    

def groAtom(a):
    # In PDB files, there might by an insertion code. To handle this, we internally add
    # constant to all resids. To be consistent, we have to do the same for gro files.
    # 32 equal ord(' '), eg an empty insertion code
    constant = 32<<20
    #012345678901234567890123456789012345678901234567890
    #    1PRN      N    1   4.168  11.132   5.291
    ## ===> atom name,        res name,          res id,    chain,
    return (a[10:15].strip(), a[5:10].strip(),   int(a[:5])+constant, " ", 
    ##               x,                 y,                 z,  atomid      
    10*float(a[20:28]),10*float(a[28:36]),10*float(a[36:44]), int(a[15:20]) )

# Simple GRO iterator
def groFrameIterator(streamIterator):
    while True:
        try:
            title = streamIterator.next()
        except StopIteration:
            break
        natoms = streamIterator.next().strip()
        if not natoms:
            break
        natoms = int(natoms)
        atoms  = [groAtom(streamIterator.next())  for i in range(natoms)] 
        yield title, atoms

#----+-------------+
## C | GENERAL I/O |
#----+-------------+

# *NOTE*: This should probably be a CheckableStream class that
# reads in lines until either of a set of specified conditions
# is met, then setting the type and from thereon functioning as
# a normal stream.
def streamTag(stream):
    # Tag the stream with the type of structure file
    # If necessary, open the stream, taking care of 
    # opening using gzip for gzipped files

    # First check whether we have have an open stream or a file
    # If it's a file, check whether it's zipped and open it
    if type(stream) == str:
        if stream.endswith("gz"):
            logging.info('Read input structure from zipped file.')
            s = gzip.open(stream)
        else:
            logging.info('Read input structure from file.')
            s = open(stream)
    else:
        logging.info('Read input structure from command-line')
        s = stream

    # Read a few lines, but save them
    x = [s.readline(), s.readline()]
    if x[-1].strip().isdigit():
        # Must be a GRO file
        logging.info("Input structure is a GRO file. Chains will be labeled consecutively.")
        yield "GRO"
    else:
        # Must be a PDB file then
        # Could wind further to see if we encounter an "ATOM" record
        logging.info("Input structure is a PDB file.")
        yield "PDB"
    
    # Hand over the lines that were stored
    for i in x:
        yield i

    # Now give the rest of the lines from the stream
    for i in s:
        yield i


#----+-----------------+
## D | STRUCTURE STUFF |
#----+-----------------+


# This list allows to retrieve atoms based on the name or the index
# If standard, dictionary type indexing is used, only exact matches are
# returned. Alternatively, partial matching can be achieved by setting
# a second 'True' argument. 
class Residue(list):
    def __getitem__(self,tag): 
        if type(tag) == int:
            # Call the parent class __getitem__
            return list.__getitem__(self,tag)
        if type(tag) == str:
            for i in self:
                if i[0] == tag:
                    return i
            else:
                return 
        if tag[1]:
            return [i for i in self if tag[0] in i[0]] # Return partial matches
        else:
            return [i for i in self if i[0] == tag[0]] # Return exact matches only


def residues(atomList):
    residue = [atomList[0]]
    for atom in atomList[1:]:
        if (atom[1] == residue[-1][1] and # Residue name check
            atom[2] == residue[-1][2] and # Residue id check
            atom[3] == residue[-1][3]):   # Chain id check
            residue.append(atom)
        else:
            yield Residue(residue)
            residue = [atom]
    yield Residue(residue)


def residueDistance2(r1,r2):
    return min([distance2(i,j) for i in r1 for j in r2])


# CUT-OFF_CHANGE
# Increased the cut-off from 2.5 to 3.0 for DNA. Might have to be adjusted for other DNA structures.
# Not guaranteed to work for proteins, recommended to use the regular martinize for them. 
def breaks(residuelist,selection=("N","CA","C","P","C2'","C3'","O3'","C4'","C5'","O5'", "OP1", "OP2"),cutoff=3.0):
    # Extract backbone atoms coordinates
    bb = [[atom[4:] for atom in residue if atom[0] in selection] for residue in residuelist]
    # Needed to remove waters residues from mixed residues.
    bb = [res for res in bb if res != []]

    # We cannot rely on some standard order for the backbone atoms.
    # Therefore breaks are inferred from the minimal distance between
    # backbone atoms from adjacent residues.
    return [ i+1 for i in range(len(bb)-1) if residueDistance2(bb[i],bb[i+1]) > cutoff]


###################################################


## !! NOTE !! ##
## XXX The chain class needs to be simplified by extracting things to separate functions/classes
class Chain:
    # Attributes defining a chain
    # When copying a chain, or slicing, the attributes in this list have to
    # be handled accordingly.
    _attributes = ("residues","sequence","seq")

    def __init__(self,options,residuelist=[],name=None):
        self.residues   = residuelist
        self._atoms     = [atom[:3] for residue in residuelist for atom in residue]
        self.sequence   = [residue[0][1] for residue in residuelist]
        # *NOTE*: Check for unknown residues and remove them if requested
        #         before proceeding.
        self.seq        = "".join([AA321.get(i,"X") for i in self.sequence])
        self.mapping    = []
        self.mappingH    = []
        self.options    = options

        # Unknown residues
        self.unknowns   = "X" in self.seq

        # Determine the type of chain
        self._type      = ""
        self.type()

        # Determine number of atoms
        self.natoms     = len(self._atoms) 

        # BREAKS: List of indices of residues where a new fragment starts
        # Only when polymeric (protein, DNA, RNA, ...)
        # For now, let's remove it for the Nucleic acids...
        self.breaks     = self.type() in ("Protein","Mixed") and breaks(self.residues) or []

        # Chain identifier; try to read from residue definition if no name is given
        self.id         = name or residuelist and residuelist[0][0][3] or ""

        # Container for coarse grained beads
        self._cg        = None

        
    def __len__(self):
        # Return the number of residues
        # DNA/RNA contain non-CAP d/r to indicate type. We remove those first.
        return len(''.join(i for i in self.seq if i.isupper()))

    def __add__(self,other):
        newchain = Chain(name=self.id+"+"+other.id)
        # Combine the chain items that can be simply added
        for attr in self._attributes:
            setattr(newchain, attr, getattr(self,attr) + getattr(other,attr))
        # Set chain items, shifting the residue numbers
        shift  = len(self)
        newchain.breaks     = self.breaks + [shift] + [i+shift for i in other.breaks]
        newchain.natoms     = len(newchain.atoms())
        # Return the merged chain
        return newchain

    def __eq__(self,other):
        return (self.seq        == other.seq    and 
                self.breaks     == other.breaks )

    # Extract a residue by number or the list of residues of a given type
    # This facilitates selecting residues for links, like chain["CYS"]
    def __getitem__(self,other):
        if type(other) == str:
            if not other in self.sequence:
                return []
            return [i for i in self.residues if i[0][1] == other]
        elif type(other) == tuple:
            # This functionality is set up for links
            # between coarse grained beads. So these are
            # checked first,
            for i in self.cg():
                if other == i[:4]:
                    return i
            else:
                for i in self.atoms():
                    if other[:3] == i[:3]:
                        return i
                else:
                    return []
        return self.sequence[other]

    # Extract a piece of a chain as a new chain
    def __getslice__(self,i,j):
        newchain = Chain(self.options,name=self.id)        
        # Extract the slices from all lists
        for attr in self._attributes:           
            setattr(newchain, attr, getattr(self,attr)[i:j])
        # Breaks that fall within the start and end of this chain need to be passed on.
        # Residue numbering is increased by 20 bits!!
        # XXX I don't know if this works.
        ch_sta,ch_end = newchain.residues[0][0][2],newchain.residues[-1][0][2]
        newchain.breaks     = [crack for crack in self.breaks if ch_sta < (crack<<20) < ch_end]
        newchain.natoms     = len(newchain.atoms())
        newchain.type()
        # Return the chain slice
        return newchain

    def _contains(self,atomlist,atom):
        atnm,resn,resi,chn = atom
        
        # If the chain does not match, bail out
        if chn != self.id:
            return False

        # Check if the whole tuple is in
        if atnm and resn and resi:
            return (atnm,resn,resi) in self.atoms()

        # Fetch atoms with matching residue id
        match = (not resi) and atomlist or [j for j in atomlist if j[2] == resi]
        if not match:
            return False

        # Select atoms with matching residue name
        match = (not resn) and match or [j for j in match if j[1] == resn]
        if not match:
            return False

        # Check whether the atom is given and listed
        if not atnm or [j for j in match if j[0] == atnm]:
            return True

        # It just is not in the list!
        return False

    def __contains__(self,other):
        return self._contains(self.atoms(),other) or self._contains(self.cg(),other)

    def __hash__(self):
        return id(self)

    def atoms(self):
        if not self._atoms:
            self._atoms = [atom[:3] for residue in self.residues for atom in residue]
        return self._atoms

    # Split a chain based on residue types; each subchain can have only one type
    def split(self):
        chains = []
        chainStart = 0
        for i in range(len(self.sequence)-1):
            if residueTypes.get(self.sequence[i],"Unknown") != residueTypes.get(self.sequence[i+1],"Unknown"):
                # Use the __getslice__ method to take a part of the chain.
                chains.append(self[chainStart:i+1])
                chainStart = i+1
        if chains:
            logging.debug('Splitting chain %s in %s chains'%(self.id,len(chains)+1))
        return chains + [self[chainStart:]]

    def getname(self,basename=None):
        name = []
        if basename:                      name.append(basename)
        if self.type() and not basename:  name.append(self.type())
        if type(self.id) == int:
            name.append(chr(64+self.id))
        elif self.id.strip():               
            name.append(str(self.id))
        return "_".join(name)

    def type(self,other=None):
        if other:
            self._type = other
        elif not self._type and len(self):
            # Determine the type of chain
            self._type     = set([residueTypes.get(i,"Unknown") for i in set(self.sequence)])
            self._type     = list(self._type)[0]
        return self._type


    # XXX The following (at least the greater part of it) should be made a separate function, put under "MAPPING"
    def cg(self,qval,force=False,com=False,dna=False):
        # Generate the coarse grained structure
        # Set the b-factor field to something that reflects the secondary structure
        
        # If the coarse grained structure is set already, just return, 
        # unless regeneration is forced.
        if self._cg and not force:
            return {'cg':self._cg, 'ff':self._FF, 'cg_NOTinb': self._NotInBcg, 'ff_NOTinb': self._NotInBff }
        self._cg = []
        atid     = 1
        bb       = [1]
        fail     = False
        previous = ''
        self._FF = []
        self._NotInBcg = [] # contains atom not in beads accorging to CG mapping
        self._NotInBff = [] # contains atom not in beads accorging to FF mapping (mappingH)


        # compute Atomic Form Factor for different qval
        atomicff = { }
        atomicff['H']=FormFactor('H',qval)
        atomicff['C']=FormFactor('C',qval)
        atomicff['N']=FormFactor('N',qval)
        atomicff['O']=FormFactor('O',qval)
        atomicff['P']=FormFactor('P',qval)
        atomicff['S']=FormFactor('S',qval)

        for residue,resname in zip(self.residues,self.sequence):
            # For DNA we need to get the O3' to the following residue when calculating COM
            # The force and com options ensure that this part does not affect itp generation or anything else
            if com:
                # Just an initialization, this should complain if it isn't updated in the loop
                store = 0
                for ind, i in enumerate(residue):
                    if i[0] == "O3'":
                        if previous != '':
                            residue[ind] = previous
                            previous = i
                        else:
                            store = ind
                            previous = i
                # We couldn't remove the O3' from the 5' end residue during the loop so we do it now
                if store > 0:
                    del residue[store]

            # Check if residues names has changed, for example because user has set residues interactively.
            residue = [(atom[0],resname)+atom[2:] for atom in residue]
            if residue[0][1] in ("SOL","HOH","TIP"):
                continue
            if not residue[0][1] in CoarseGrained.mapping.keys():
                logging.warning("Skipped unknown residue %s\n"%residue[0][1])
                continue
            if not residue[0][1] in CoarseGrained.mappingH.keys():
                logging.warning("Skipped unknown residue %s in mappingH\n"%residue[0][1])
                continue

            # Get the mapping for this residue
            # CG.map returns bead coordinates and mapped atoms
            # This will fail if there are (too many) atoms missing, which is
            # only problematic if a mapped structure is written; the topology
            # is inferred from the sequence. So this is the best place to raise 
            # an error
            NEWMAP=mapCG(residue)
            try:
                beads, ids, atm, aid = NEWMAP['pos']
                beads      = zip(CoarseGrained.names[residue[0][1]],beads,ids,atm,aid)

            except ValueError:
                logging.error("Too many atoms missing from residue %s %d(ch:%s):",residue[0][1],residue[0][2]>>20,residue[0][3])
                logging.error(repr([ i[0] for i in residue ]))
                fail = True

            for name,(x,y,z),ids,atm,aid in beads:                    
                # Add the bead with coordinates and secondary structure id to the list
                self._cg.append((name,residue[0][1][:3],residue[0][2],residue[0][3],x,y,z,atm,aid))
                # Add the ids to the list, after converting them to indices to the list of atoms
                #self.mapping.append([atid+i for i in ids])
           
            self._NotInBcg.append((residue[0][1][:3],residue[0][2],residue[0][3],NEWMAP['NOTinb']))
            

            NEWMAPH=mapFF(residue,qval,atomicff)
            try:
                beadsH, idsH, atmH, aidH = NEWMAPH['ff']
                beadsH      = zip(CoarseGrained.names[residue[0][1]],beadsH,idsH,atmH,aidH)

            except ValueError:
                logging.error("Too many atoms missing for Mapping H from residue %s %d(ch:%s):",residue[0][1],residue[0][2]>>20,residue[0][3])
                logging.error(repr([ i[0] for i in residue ]))
                fail = True

            for name,beadff,ids,atm,aid in beadsH:                    
                # Add the bead with coordinates and secondary structure id to the list
                self._FF.append((name,residue[0][1][:3],residue[0][2],residue[0][3],beadff,atm,aid))
                # Add the ids to the list, after converting them to indices to the list of atoms
                #self.mappingH.append([atid+i for i in ids])
           
            self._NotInBff.append((residue[0][1][:3],residue[0][2],residue[0][3],NEWMAPH['NOTinb']))

            # Increment the atom id; This pertains to the atoms that are included in the output.
            atid += len(residue)

            # Keep track of the numbers for CONECTing
            bb.append(bb[-1]+len(beads))

        if fail:
            logging.error("Unable to generate coarse grained structure due to missing atoms.")
            sys.exit(1)

        return {'cg':self._cg, 'ff':self._FF, 'cg_NOTinb': self._NotInBcg, 'ff_NOTinb': self._NotInBff}

 

#############
## 8 # MAIN #  -> @MAIN <-
#############
import sys,logging,random,math,os,re
import numpy as np

def main(options,qval):
    # Check whether to read from a gro/pdb file or from stdin
    # We use an iterator to wrap around the stream to allow
    # inferring the file type, without consuming lines already
    inStream = streamTag(options["-f"] and options["-f"].value or sys.stdin)
    

    # The streamTag iterator first yields the file type, which 
    # is used to specify the function for reading frames
    fileType = inStream.next()
    if fileType == "GRO":
        frameIterator = groFrameIterator
    else:
        frameIterator = pdbFrameIterator
        # in PDB hydrogens could be named differently -> Much better to use gro files!
        logging.error("In PDB hydrogens could be named differently. Better to use gro files as input!")
        sys.exit(1)
    

    ## ITERATE OVER FRAMES IN STRUCTURE FILE ##

    # Now iterate over the frames in the stream
    # This should become a StructureFile class with a nice .next method
    model     = 1
    cgOutPDB  = None
    cgOutFF = None
    cgOutIND = None
    ffOutIND = None
    cgOutEXC = None
    ffOutEXC = None

    for title,atoms in frameIterator(inStream):
    
        if fileType == "PDB":
            # The PDB file can have chains, in which case we list and process them specifically
            # TER statements are also interpreted as chain separators
            # A chain may have breaks in which case the breaking residues are flagged
            chains = [ Chain(options,[i for i in residues(chain)]) for chain in pdbChains(atoms) ]    
        else:
            # The GRO file does not define chains. Here breaks in the backbone are
            # interpreted as chain separators. 
            residuelist = [residue for residue in residues(atoms)]
            # The breaks are indices to residues
            broken = breaks(residuelist)
            # Reorder, such that each chain is specified with (i,j,k)
            # where i and j are the start and end of the chain, and 
            # k is a chain identifier
            chains = zip([0]+broken,broken+[len(residuelist)],range(len(broken)+1))
            chains = [ Chain(options,residuelist[i:j],name=chr(65+k)) for i,j,k in chains ]
    
        # Check the chain identifiers
        if model == 1 and len(chains) != len(set([i.id for i in chains])):
            # Ending down here means that non-consecutive blocks of atoms in the 
            # PDB file have the same chain ID. The warning pertains to PDB files only, 
            # since chains from GRO files get a unique chain identifier assigned.
            logging.warning("Several chains have identical chain identifiers in the PDB file.")
   
        # Check if chains are of mixed type. If so, split them.
        # Note that in some cases HETATM residues are part of a 
        # chain. This will get problematic. But we cannot cover
        # all, probably.
        demixedChains = []
        for chain in chains:
            demixedChains.extend(chain.split())
        chains = demixedChains

        n = 1
        logging.info("Found %d chains:"%len(chains))
        for chain in chains:
            logging.info("  %2d:   %s (%s), %d atoms in %d residues."%(n,chain.id,chain._type,chain.natoms,len(chain)))
            n += 1
    
        # Check all chains
        keep = []
        for chain in chains:
            if chain.type() == "Water":
                logging.info("Removing %d water molecules (chain %s)."%(len(chain),chain.id))
            elif chain.type() in ("Protein","Nucleic"):
                keep.append(chain)
            else:
                logging.info("Removing HETATM chain %s consisting of %d residues."%(chain.id,len(chain)))
        chains = keep


        if model == 1:
            order = []
            order.extend([j for j in range(len(chains)) if not j in order])

        # Get the total length of the sequence
        seqlength = sum([len(chain) for chain in chains])
        logging.info('Total size of the system: %s residues.'%seqlength)
    
            
    
        # Write the coarse grained structure if requested
        if options["-x"].value:
            logging.info("Writing coarse grained structure and form factor.")
            if cgOutPDB == None:
                cgOutPDB = open(options["-x"].value,"w")
            cgOutPDB.write("MODEL %8d\n"%model)
            cgOutPDB.write(title)
            if cgOutIND == None:
                cgOutIND = open(options["-xind"].value,"w")
                cgOutIND.write('# IDS: resID beadID resID chainID atomsIDS \n# NAME: resNAME beadNAME atomsNAME \n')
            if cgOutEXC == None:
                cgOutEXC = open(options["-xexc"].value,"w")
                cgOutEXC.write('# resNAME atomNAME chainID resID atomID\n')

            if cgOutFF == None:
                cgOutFF = open(options["-ff"].value,"w")
            cgOutFF.write("# pdbID beadID resNAME beadNAME [ beadFF for qvalues ]\npdbID beadID resNAME beadNAME\t")
            np.savetxt(cgOutFF, qval,newline=' ',fmt="%.3f")
            cgOutFF.write('\n')
            
            if ffOutIND == None:
                ffOutIND = open(options["-ffind"].value,"w")
                ffOutIND.write('# IDS: resID beadID resID chainID pdbID atomsIDS\n# NAME: resNAME beadNAME atomsNAME \n')
            if ffOutEXC == None:
                ffOutEXC = open(options["-ffexc"].value,"w")
                ffOutEXC.write('# resNAME atomNAME chainID resID atomID pdbID\n')

            cgMAP = open("cg.map","w")
           
            atid = 1; atidff = 1;
            write_start = 0
            for i in order:
                ci = chains[i]
                coarseGrained = ci.cg(qval,com=True)
                if coarseGrained:
                    # For DNA we need to remove the first bead on the 5' end and shift the atids. 
                    if ci.type() == 'Nucleic':
                        write_start = 1
                    else:
                        write_start = 0

                    # CG pdb and index
                    for name,resn,resi,chain,x,y,z,atm,aid in coarseGrained['cg'][write_start:]:
                        insc  = resi>>20
                        resi -= insc<<20
                        cgOutPDB.write(pdbAtomLine%(atid,name,resn[:3],chain,resi,chr(insc),x,y,z,1,0))
                        
                        #cgOutIND.write
                        line1 = 'IDS: %4d%1s %5d %i-%1s'%(resi,chr(insc),atid,i,chain)
                        line2 = 'NAME: %4s %4s '%(resn[:3],name)
                        linemap = 'bead' + str(atid) + ': CENTER ATOMS='
                        for k in aid:
                            line1 += '%5d '%k 
                            linemap+='%d,'%k
                        linemap = linemap[:-1] + ' WEIGHTS='
#                        linew=CoarseGrained.mass.get(i[0][0],0)

                        for k in atm:
                            line2 += '%4s '%k
                            linemap+='%s,'%CoarseGrained.mass.get(k[0][0],0)

                        line1 += '\n'
                        line2 += '\n'
                        cgOutIND.write(line1)
                        cgOutIND.write(line2)

                        linemap = linemap[:-1] + '\n'
                        cgMAP.write(linemap)
                        
                        atid += 1
                    cgOutPDB.write("TER\n") 
                     
                    # CG excluded atoms
                    if len(coarseGrained['cg_NOTinb'][0]) > 0: 
                        for resn,resi,chain,atoms in coarseGrained['cg_NOTinb'][write_start:]:
                            insc  = resi>>20
                            resi -= insc<<20
                            for k in atoms:
                                cgOutEXC.write('%4s %4s %i-%s %4d%1s %5d\n'%(resn[:3],k[0],i,chain,resi,chr(insc),k[1]))


                    # FF out and index                    
                    for name,resn,resi,chain,beadff,atm,aid in coarseGrained['ff'][write_start:]:
                        insc  = resi>>20
                        resi -= insc<<20
                        cgOutFF.write('%-12s %5d %4s %4s\t' %(options["-fname"].value,atidff,resn[:3],name))
                        np.savetxt(cgOutFF,beadff,newline=' ',fmt="%.3f")
                        cgOutFF.write('\n')
                        #ffOutIND.write
                        line1 = 'IDS: %4d%1s %5d %i-%1s %-12s'%(resi,chr(insc),atidff,i,chain,options["-fname"].value)
                        line2 = 'NAME: %4s %4s '%(resn[:3],name)
                        for k in aid:
                            line1 += '%5d '%k 
                        for k in atm:
                            line2 += '%4s '%k 
                        line1 += '\n'
                        line2 += '\n'
                        ffOutIND.write(line1)
                        ffOutIND.write(line2)

                        atidff += 1

                    # FF excluded atoms
                    if len(coarseGrained['ff_NOTinb'][0]) > 0: 
                        for resn,resi,chain,atoms in coarseGrained['ff_NOTinb'][write_start:]:
                            insc  = resi>>20
                            resi -= insc<<20
                            for k in atoms:
                                ffOutEXC.write('%4s %4s %i-%s %4d%1s %8d %12s\n'%(resn[:3],k[0],i,chain,resi,chr(insc),k[1],options["-fname"].value))            

                else:
                    logging.warning("No mapping for coarse graining chain %s (%s); chain is skipped."%(ci.id,ci.type()))
            cgOutPDB.write("ENDMDL\n")
            cgOutPDB.close();  cgOutIND.close(); cgOutEXC.close()
            cgOutFF.close(); ffOutIND.close(); ffOutEXC.close()
 
        model += 1


def compare(a,b):
    return len(a)==len(b) and len(a)==sum([1 for i in a if i in b]) and len(b)==sum([1 for i in b if i in a])


def ComputeAverages(fin,findex,foutav,foutfit):
    
    IDS=np.genfromtxt(fin,usecols=(0,1,2,3),skip_header=2,dtype={'names': ('pdbID','beadID', 'resNAME', 'beadNAME'),'formats': ('S10','d', 'S10', 'S10')})

    alldata=np.genfromtxt(fin)
    formfactor=alldata[1:,4:]
    qval=alldata[0,4:]

    atomIDS=[]; 
    j=0
    for line in open(findex):
        if "NAME:" in line:
            if "#" not in line:
                atomIDS.append(line.splitlines()[0].split()[1:])
                j += 1

    if len(atomIDS) == len(IDS):
        #beadTYPESold=np.unique(atomIDS)
        beadTYPES = [];
        for x in atomIDS:
            a=0
            if(len(beadTYPES)==0):
                beadTYPES.append(x)
            else:
                for y in beadTYPES:
                    if(compare(x,y)==True):
                        a+=1
                if a==0:
                    beadTYPES.append(x)

        #print(len(beadTYPES),len(beadTYPESold))
        
        ind = [];
        for t in beadTYPES:
            tmp=[]; #pdb=[]; 
            for j in range(len(atomIDS)):
                if compare(t,atomIDS[j]):
                    tmp.append(j)
                    #pdb.append(IDS['pdbID'][j])
            ind.append((t[0],t[1],np.array(tmp)))
        
        f1 = open(foutav,"w")
        f1.write('%4s %4s %4s\t' %("Res","Bead","N"))
        np.savetxt(f1,qval,newline=' ',fmt="%.3f")

        f2 = open(foutfit,"w")
        f2.write('%4s %4s %4s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\t' %("Res","Bead","N","A0","A1","A2","A3","A4","A5","A6"))

        onev=np.ones(7);
        lowb=np.multiply(onev,-np.inf);     
        upb=np.multiply(onev,np.inf);    
        for i in range(len(ind)):
            beadTYPES[i].append(str(len(formfactor[ind[i][2]])))
            title='_'.join(beadTYPES[i]) + ".log"
            fb = open(title,"w")
            for k in range(len(formfactor[ind[i][2]])):
                if k==0:
                    fb.write('%4s %4s %7s %7d\t' %(ind[i][0],ind[i][1],IDS['pdbID'][ind[i][2][k]],int(IDS['beadID'][ind[i][2][k]])))
                else:
                    fb.write('\n%4s %4s %7s %7d\t' %(ind[i][0],ind[i][1],IDS['pdbID'][ind[i][2][k]],int(IDS['beadID'][ind[i][2][k]])))
                np.savetxt(fb,formfactor[ind[i][2]][k],newline=' ',fmt="%.3f")
            fb.close()
            av=np.average(formfactor[ind[i][2]],axis=0)
            f1.write('\n%4s %4s %4d\t' %(ind[i][0],ind[i][1],len(ind[i][2])))
            np.savetxt(f1,av,newline=' ',fmt="%.3f")
            f2.write('\n%4s %4s %4d\t' %(ind[i][0],ind[i][1],len(ind[i][2])))

            lowb[0]=av[0];
            upb[0]=av[0]+np.finfo(np.float32).eps;
            popt, pcov = curve_fit(func6, qval, av, bounds=(lowb,upb))
            np.savetxt(f2,popt,newline=' ',fmt="%-3.6f")

        f1.close()
        f2.close()
    else:
        print("Lines number is different in {} and {}. Something went wrong probably. Double check and perform the analysis manually!".format(fin,findex))


if __name__ == '__main__':
    import sys,logging,time

    start = time.time()
    stloc = time.localtime(start)
    logging.info("Start at time: {}:{}:{} of {}/{}/{}.".format(stloc[3],stloc[4],stloc[5],stloc[2],stloc[1],stloc[0]))

    args = sys.argv[1:]
    if not args:
        logging.error("NO INPUT!")
        sys.exit(1)

    # DEFINE list of qvalue
    qval=np.arange(0.000,2.001,0.002)   # U.M.: ANG-1
    #qval=np.arange(0.000,2.001,0.01)

    options = options
    try:
        b=args.index('-filelist')
    except ValueError:
        options = option_parser(args,options)
        fname="{}".format(options["-f"].value)
        options['-fname'] =  Option(str,1,None)
        options['-fname'].setvalue([fname[:4] for i in range(options['-fname'].num)])
        main(options,qval)
    else:
        filelist=args[b+1]
        with open(filelist) as f:
            files = f.read().splitlines()
        
        count = 0
        cgOutFF_list = []; ffOutIND_list = []; ffOutEXC_list = [];
        for f in files:
            fname="{}_{}".format(count,f[:4])
            logging.info("\n\nREADING INPUT FILE: {}".format(f))
            cgOutFF_list.append(str("ff_{}.out".format(fname)))
            ffOutIND_list.append(str("ffInd_{}.out".format(fname)))
            ffOutEXC_list.append(str("ffExcl_{}.out".format(fname)))
            myfile = './{}'.format(cgOutFF_list[-1])
            if os.path.isfile(myfile):
                print("FILE {} already exists!".format(cgOutFF_list[-1]))
            else:
                myoptions = option_parser2(f,fname,options)
                main(myoptions,qval)
            
            count += 1

        # Concatenate files
        cgOutFF = open("ff.out","w")
        cgOutFF.write("# pdbID beadID resNAME beadNAME [ beadFF for qvalues ]\npdbID beadID resNAME beadNAME\t")
        np.savetxt(cgOutFF, qval,newline=' ',fmt="%.3f")
        cgOutFF.write('\n')
        concatFiles(cgOutFF,cgOutFF_list,"pdbID")

        ffOutIND = open("ffInd.out","w")
        ffOutIND.write('# IDS: resID beadID resID chainID pdbID atomsIDS\n# NAME: resNAME beadNAME atomsNAME \n')
        concatFiles(ffOutIND,ffOutIND_list,"bead")
        
        ffOutEXC = open("ffExcl.out","w")
        ffOutEXC.write('# resNAME atomNAME chainID resID atomID pdbID\n')
        concatFiles(ffOutEXC,ffOutEXC_list,"resNAME")

        cgOutFF.close(); ffOutIND.close(); ffOutEXC.close();

        # Compute Averages
        ComputeAverages("ff.out","ffInd.out","ff_average.out","ff_averfit.out")
        
    stop = time.time()
    stoploc = time.localtime(stop)
    logging.info("\n\nEnded at time: {}:{}:{} of {}/{}/{}.".format(stoploc[3],stoploc[4],stoploc[5],stoploc[2],stoploc[1],stoploc[0]))
    logging.info("Total time needed: {} sec\n".format(stop-start))
