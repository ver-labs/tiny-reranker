"""Focused proof that graph capture precedes evaluation and kernels execute."""
import argparse
from .inference import Inference
from .common import HERE,fixtures,pairs,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='mps');a=p.parse_args();report={}
    for path in ('mlx','mlx_compiled','metal','metal_compiled'):
        model=Inference(a.device,path);report[path]=model.dispatch(pairs(fixtures()[0],16));print(path,report[path]['custom_sites'],report[path]['observed_math_attention_softmax_sites'],flush=True)
    save(HERE/f'{a.device}_dispatch_proof.json',report)

if __name__=='__main__':main()
