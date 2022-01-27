#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle

import math
from collections import defaultdict

import os.path as op

from selfies import encoder, decoder
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import Descriptors


_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    
    score1 /= (nf + 1e-7)

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def processMols(mols):
    print('smiles\tName\tsa_score')
    for i, m in enumerate(mols):
        if m is None:
            continue

        s = calculateScore(m)

        smiles = Chem.MolToSmiles(m)
        print(smiles + "\t" + m.GetProp('_Name') + "\t%3f" % s)




def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
         
    
def get_logP(mol):
    '''Calculate logP of a molecule 
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for which logP is to calculates
    
    Returns:
    float : logP of molecule (mol)
    '''
    return Descriptors.MolLogP(mol)


def get_SA(mol):
    return calculateScore(mol)

def calc_RingP(mol):
    '''Calculate Ring penalty for each molecule in unseen_smile_ls,
       results are recorded in locked dictionary props_collect 
    '''
    cycle_list = mol.GetRingInfo().AtomRings() 
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def calculate_pLogP(smiles):  
    mol, smiles_canon, done = sanitize_smiles(smiles)
    logP_scores   = [get_logP(mol)]
    SA_scores     = [get_SA(mol)]
    RingP_scores  = [calc_RingP(mol)]

    #logP_norm  = np.array([((x - 2.4729421499641497) / 1.4157879815362406) for x in logP_scores])
    #SAS_norm   = np.array([((x - 3.0470797085649894) / 0.830643172314514) for x in SA_scores])
    #RingP_norm = [((x - 0.038131530820234766) / 0.2240274735210179) for x in RingP_scores]

    logP_norm  = (np.array(logP_scores) - 2.4729421499641497) / 1.4157879815362406
    SAS_norm   = (np.array(SA_scores)  - 3.0470797085649894) / 0.830643172314514
    RingP_norm = (np.array(RingP_scores)  - 0.038131530820234766) / 0.2240274735210179

    return logP_norm[0] - SAS_norm[0] - RingP_norm[0] 





import helper
if __name__=='__main__':
    '''
    benzene = 'C1=CC=CC=C1'
    mol = Chem.MolFromSmiles(benzene)
    print('Calc: ', calculateScore(mol))


    A = 'C1CCN(CC1)CCOC2=CC=C(C=C2)C(=O)C3=C(SC4=C3C=CC(=C4)O)C5=CC=C(C=C5)O'

    A = 'CNC1=CC(=CC=C1C1=NC2=NC(=NC3=CC=CC(=N1)N23)C1=C(NC)C=C(C=C1)N(C1=CC=CC=C1)C1=CC=CC=C1)N(C1=CC=CC=C1)C1=CC=CC=C1'
    '''
    '''
    for i in range(35):
        counter = 35 - i
        A1 = '[C]' * counter
        A2 = '[S]' * i
        A = A1 + A2
        A = decoder(A)
        
        print(A)
        #print(len(A))

        mol, smiles_canon, done = sanitize_smiles(A)
        logP_scores   = [get_logP(mol)]
        SA_scores     = [get_SA(mol)]
        RingP_scores  = [calc_RingP(mol)]
        QED_score     = np.array([Chem.QED.qed(mol)])

        #print('QED: ', QED_score)
        #print('SAS: ', SA_scores)
        #print('RingP: ', RingP_scores)
        #print('logP: ', logP_scores)
        #print('plogp: ', calculate_pLogP(A))



        #print('len: ', len(A[0]))
        #print('QED: ', QED_score)
        #print('NON-normalized logP = ', logP_scores[0] - SA_scores[0] - RingP_scores[0])
    

        #logP_norm  = np.array([((x - 2.4729421499641497) / 1.4157879815362406) for x in logP_scores])
        #SAS_norm   = np.array([((x - 3.0470797085649894) / 0.830643172314514) for x in SA_scores])
        #RingP_norm = [((x - 0.038131530820234766) / 0.2240274735210179) for x in RingP_scores]

        #print('Normalized logP= ', logP_norm[0] - SAS_norm[0] - RingP_norm[0])

        print('####################')
        B = '[C]' * 35
        B = decoder(B)
        
        #print(B)
        #print(len(B))

        mol, smiles_canon, done = sanitize_smiles(B)
        logP_scores   = [get_logP(mol)]
        SA_scores     = [get_SA(mol)]
        RingP_scores  = [calc_RingP(mol)]
        QED_score     = np.array([Chem.QED.qed(mol)])

        #print('QED: ', QED_score)
        #print('SAS: ', SA_scores)
        #print('RingP: ', RingP_scores)
        #print('logP: ', logP_scores)
        #print('plogp: ', calculate_pLogP(B))

        print(calculate_pLogP(A))
        
        if calculate_pLogP(B) > calculate_pLogP(A):
            print('############################ ', i)
            print(calculate_pLogP(B))
            print(calculate_pLogP(A))
        '''    

    
    B = '[S]' * 81
    B = decoder(B)
    
    #print(B)
    #print(len(B))

    mol, smiles_canon, done = sanitize_smiles(B)
    logP_scores   = [get_logP(mol)]
    SA_scores     = [get_SA(mol)]
    RingP_scores  = [calc_RingP(mol)]
    QED_score     = np.array([Chem.QED.qed(mol)])

    #print('QED: ', QED_score)
    #print('SAS: ', SA_scores)
    #print('RingP: ', RingP_scores)
    #print('logP: ', logP_scores)
    #print('plogp: ', calculate_pLogP(B))

    print(calculate_pLogP(B))


    #
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
