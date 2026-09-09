import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_size_binned as b

class BinnedTests(unittest.TestCase):
    def test_boundaries(self):
        np.testing.assert_array_equal(b.assign_bins(np.array([22,499.99,500,999.99,1000,2000,10000,50000,485109]),[500,1000,2000,10000,50000]),[0,0,1,1,2,3,4,5,5])
    def test_held_out_targets_cannot_change_fit(self):
        rng=np.random.default_rng(3);n=25
        features=rng.random((n,12,3));parcel=np.arange(n);support=parcel.copy();area=np.full(n,600.);y=rng.uniform(0,10,(n,12));folds=np.arange(n)%5+1
        cfg={'alphas':[.1,1,10,100,1000],'inner_folds':5,'seed':20260903}
        first=b.fit_fold(cfg,features,parcel,support,area,y,folds,1)
        changed=y.copy();changed[folds==1]=999999
        second=b.fit_fold(cfg,features,parcel,support,area,changed,folds,1)
        for i in [1,2,3,4,5]:np.testing.assert_array_equal(first[i],second[i])
        self.assertEqual(first[6],second[6])

if __name__=='__main__':unittest.main()
