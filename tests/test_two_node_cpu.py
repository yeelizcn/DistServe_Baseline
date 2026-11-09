"""CPU tests of conversion and deployment contracts, no compiled backend required."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class PlacementContract(unittest.TestCase):
    def setUp(self):
        self.m = load('cross_node', 'distserve/cross_node.py')
        self.p = types.SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1)
        self.cfg = dict(prefill_node_id='p', decode_node_id='d',
                        init_method='tcp://10.0.0.1:12345', timeout_s=10)
    def test_valid(self):
        self.m.validate_config(self.cfg,self.p,self.p)
    def test_same_node_rejected(self):
        self.cfg['decode_node_id']='p'
        with self.assertRaises(ValueError): self.m.validate_config(self.cfg,self.p,self.p)
    def test_mismatched_parallel_rejected(self):
        with self.assertRaises(ValueError):
            self.m.validate_config(self.cfg,self.p,types.SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1))
    def test_missing_endpoint_rejected(self):
        self.cfg['init_method']=None
        with self.assertRaises(ValueError): self.m.validate_config(self.cfg,self.p,self.p)

class QwenConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        logger=types.ModuleType('distserve.logger')
        import logging
        logger.init_logger=logging.getLogger
        sys.modules['distserve.logger']=logger
        cls.m=load('converter_test','distserve/downloader/converter.py')
    def test_qkv_bias_and_balanced_two_shards(self):
        t=self.torch
        weights={}
        original={}
        for kind,heads in [('q',28),('k',4),('v',4)]:
            w=t.arange(heads*128*8,dtype=t.float32).reshape(heads*128,8)
            b=t.arange(heads*128,dtype=t.float32)+(10000 if kind=='k' else 20000 if kind=='v' else 0)
            weights[f'model.layers.0.self_attn.{kind}_proj.weight']=w
            weights[f'model.layers.0.self_attn.{kind}_proj.bias']=b
            original[kind]=(w,b)
        weights['model.layers.0.self_attn.o_proj.weight']=t.zeros(8,3584)
        tensors,heads,dim=self.m.preprocess_qwen2(weights)
        self.assertEqual((heads,dim),(28,128))
        with tempfile.TemporaryDirectory() as tmp:
            self.m.divideWeightAndSave(tmp,tensors,self.m.qwen2NameTranslator,heads,dim,2)
            for rank in range(2):
                for suffix,axis in [('weight',1),('bias',0)]:
                    key=f'decoder.layers.0.self_attn.qkv_proj.{suffix}'
                    got=getattr(t.jit.load(str(Path(tmp)/f'{key}.tp_{rank}.pt')),key)
                    chunks=[]
                    for kind,n in [('q',28),('k',4),('v',4)]:
                        raw=original[kind][0 if suffix=='weight' else 1]
                        chunk=raw[rank*n*64:(rank+1)*n*64]
                        chunks.append(chunk.T if suffix=='weight' else chunk)
                    expected=t.cat(chunks,dim=axis)
                    t.testing.assert_close(got.reshape(expected.shape),expected)
    def test_missing_bias_rejected(self):
        with self.assertRaises(ValueError): self.m.preprocess_qwen2({})

if __name__=='__main__': unittest.main()
