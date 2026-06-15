import parmed as pmd
import os
import glob

amber = pmd.load_file('4DII_tleap_output.top', '4DII_tleap_output.crd')
amber.save('4DII_parmed_output.top', format='gromacs')
amber.save('4DII_parmed_output.gro')
