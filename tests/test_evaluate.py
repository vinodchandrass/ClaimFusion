import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evaluate import metrics,evaluate
class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        m=metrics(["SUPPORTS","REFUTES"],["SUPPORTS","REFUTES"])
        self.assertEqual((m["accuracy"],m["macro_f1"]),(1,1))
    def test_determinism(self):
        r=[{"claim_id":str(i),"gold":g,"frozen":f,"adapted":a} for i,(g,f,a) in enumerate([("SUPPORTS","REFUTES","SUPPORTS"),("REFUTES","REFUTES","REFUTES"),("SUPPORTS","SUPPORTS","SUPPORTS")])]
        self.assertEqual(evaluate(r,100,42),evaluate(r,100,42))
    def test_duplicate_rejected(self):
        r=[{"claim_id":"x","gold":"SUPPORTS","frozen":"SUPPORTS","adapted":"SUPPORTS"}]*2
        with self.assertRaises(ValueError):evaluate(r,100,42)
if __name__=="__main__":unittest.main()
