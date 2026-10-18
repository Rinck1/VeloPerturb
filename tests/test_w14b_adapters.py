"""New metadata/RENGE adapters preserve existing arithmetic and sample roles."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/phase0'))
from w4_arms import predict_arms
from w4_streams import SourceTransform,population_features
from w14b_renge_router import mean_anchors,mean_features
from w14b_data_download import parse_soft


def test_mean_readout_matches_frozen_w4_method():
    rng=np.random.default_rng(1)
    train=dict(state=rng.normal(size=(7,3)),vel_pool=rng.normal(size=(7,3)))
    test=dict(state=rng.normal(size=(1,3)),vel_pool=rng.normal(size=(1,3)))
    y=rng.normal(size=(7,3));et=rng.normal(size=(7,4));ev=rng.normal(size=(1,4));controls=np.array([1,0,0,0,0,0,0],dtype=bool)
    _,actual,selection=mean_anchors(train,test,y,et,ev,controls)
    expected,_=predict_arms(train,test,y,et,ev,arms=('vel_pool',),controls=controls)
    np.testing.assert_array_equal(actual,expected['vel_pool'])


def test_population_mean_features_use_same_pooling_as_w4():
    rng=np.random.default_rng(2);s=sparse.csr_matrix(rng.poisson(4,(60,12)));u=sparse.csr_matrix(rng.poisson(1,(60,12)))
    ids=np.array([f'cell{i}' for i in range(60)]);genes=np.array([f'g{i}' for i in range(12)])
    transform=SourceTransform.fit(s,u,cell_ids=ids,gene_names=genes,allowed_ids=set(ids),n_hvg=8,dimensions=3,clusters=2)
    labels=np.array(['TF1']*20+['TF2']*20+['TF3']*20);order=['TF1','TF2','TF3']
    expected=population_features(transform,s,u,labels,order,ids,neighbors=4)
    actual=mean_features(transform,s,u,labels,order)
    for key in ('state','vel_pool'):np.testing.assert_array_equal(actual[key],expected[key])


def test_soft_parser_preserves_library_identity_and_no_expression_rows():
    text='^SAMPLE = GSM1\n!Sample_title = KO_A1_GEX\n!Sample_characteristics_ch1 = treatment: untreated\n!Sample_relation = SRA: SRX1\n'
    result=parse_soft(text)
    assert result[0]['title']==['KO_A1_GEX']
    assert result[0]['characteristics']==['treatment: untreated']
