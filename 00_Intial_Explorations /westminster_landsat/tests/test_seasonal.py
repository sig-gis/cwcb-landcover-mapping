import ast
import importlib.util
import math
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('seasonal',ROOT/'scripts/analyze_seasonal.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def original_functions():
    scope = dict(np=np,math=math,nnls=nnls)
    for file,names in [('pipeline.py',['sat_scaler','ridge_nnls']),
                       ('landsat_only.py',['bottomup_design'])]:
        tree = ast.parse((ROOT/'analysis/original_src'/file).read_text())
        functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=functions,type_ignores=[]),file,'exec'),scope)
        if file=='pipeline.py':
            scope['core'] = types.SimpleNamespace(sat_scaler=scope['sat_scaler'],AREA_UNIT=100)
    return scope


class SeasonalTests(unittest.TestCase):
    def test_matches_original_design_and_nnls(self):
        old = original_functions()
        rng = np.random.default_rng(4)
        features = rng.normal(size=(5,12,3))
        p = np.array([0,0,1,1,2]); support=np.array([0,1,1,2,4]); a=np.array([40.,60.,30.,70.,500.])
        fragments=[dict(parcel=int(pi),sat=features[si],area_m2=ai) for pi,si,ai in zip(p,support,a)]
        train=np.array([0,1]); d,lo,span=s.design(features,p,support,a,train,3)
        for m in range(12):
            expected,l,h=old['bottomup_design'](fragments,3,m,train)
            np.testing.assert_allclose(d[m],expected,atol=1e-12)
            np.testing.assert_allclose(lo[m],l); np.testing.assert_allclose(span[m],h)
            y=np.array([3.,9.])
            np.testing.assert_allclose(s.solve(d[m,train],y,10),old['ridge_nnls'](expected[train],y,10))
        modified=features.copy(); modified[4]=100000
        _,lo2,span2=s.design(modified,p,support,a,train,3)
        np.testing.assert_array_equal(lo,lo2); np.testing.assert_array_equal(span,span2)

    def test_missing_account_month_not_zero(self):
        con=sqlite3.connect(':memory:')
        fields=','.join(f'cow_{m}_21 REAL' for m in s.MONTHS)
        con.execute(f'CREATE TABLE cow_accounts_joined(parcel_fid INTEGER,join_method TEXT,{fields})')
        con.execute('INSERT INTO cow_accounts_joined VALUES ('+','.join('?'*14)+')',[1,'ok',*[1.]*12])
        con.execute('INSERT INTO cow_accounts_joined VALUES ('+','.join('?'*14)+')',[1,'ok',None,*[2.]*11])
        obs=s.observations(con)
        self.assertTrue(pd.isna(obs.observed_01[0])); self.assertEqual(obs.observed_02[0],3)
        output=s.add_output_fields(obs,np.full((1,12),5.))
        self.assertTrue(pd.isna(output.sum_abs_delta[0]))
        self.assertEqual(output.delta_02[0],2)

    def test_absolute_monthly_error_is_not_absolute_annual_error(self):
        obs=pd.DataFrame({f'observed_{m:02d}':[10.] for m in range(1,13)})
        pred=np.full((1,12),10.); pred[0,0]=15; pred[0,1]=5
        result=s.add_output_fields(obs,pred)
        self.assertEqual(result.sum_abs_delta[0],10.)
        self.assertEqual(result.annual_delta[0],0.)

    def test_qa_and_monthly_medians(self):
        good=[10000,12000,20000,16000,0,0,'2021-01-05',1,'LC08_test']
        cloud=good.copy(); cloud[4]=8
        saturated=good.copy(); saturated[5]=1
        missing=good.copy(); missing[0]=None
        monthly,audit=s.reduce_observations([good,good,cloud,saturated,missing])
        self.assertEqual(monthly['clear_count'][0],1)
        self.assertEqual(audit['duplicate_alias_observations'],1)
        self.assertEqual(audit['missing_required_bands'],1)
        self.assertEqual(audit['masked_qa'],1)
        self.assertEqual(audit['masked_saturation'],1)
        b,r,n,w=np.array(good[:4])*0.0000275-.2
        np.testing.assert_allclose(monthly.select('ndvi','ndmi','evi').to_numpy()[0],
                                   [(n-r)/(n+r),(n-w)/(n+w),2.5*(n-r)/(n+6*r-7.5*b+1)])


if __name__=='__main__': unittest.main()
