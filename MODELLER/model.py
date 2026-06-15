from modeller import *

env = Environ()
aln = Alignment(env)
mdl = Model(env, file='4DII_protein')
aln.append_model(mdl, align_codes='4DII_protein', atom_files='4DII_protein.pdb')
aln.append(file='target.ali', align_codes='target')
aln.align2d(max_gap_length=50)
aln.write(file='thrombin_model2.ali', alignment_format='PIR')
aln.write(file='thrombin_model2.pap', alignment_format='PAP')
print("Wrote thrombin_model")
