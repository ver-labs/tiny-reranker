"""Size-bounded JSON-lines worker; stdout is exclusively the protocol."""
import argparse
from contextlib import redirect_stdout
import json
import sys

MAX_FRAME = 8 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['mlx','pytorch'], required=True)
    parser.add_argument('--pair-batch-size', type=int, choices=range(1,65), default=8)
    args = parser.parse_args()
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        # Import S1 before MLX to retain MLX_ENABLE_TF32=0.
        from experiments.S1.backend import Backend, CHECKPOINT, sha256
        from experiments.S1.serve import Request
        backend = Backend(args.backend)
        backend.score(['warmup'],['warmup document'])

    def send(payload):
        wire = json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')+b'\n'
        if len(wire)>MAX_FRAME:
            raise RuntimeError('Worker response exceeds frame limit')
        protocol.buffer.write(wire)
        protocol.buffer.flush()

    send({'type':'ready','protocol_version':1,'backend':args.backend,
          'pair_batch_size':args.pair_batch_size,'max_length':128,
          'checkpoint_sha256':sha256(CHECKPOINT)})
    while True:
        wire = sys.stdin.buffer.readline(MAX_FRAME+1)
        if not wire:
            break
        if len(wire)>MAX_FRAME or not wire.endswith(b'\n'):
            raise RuntimeError('Invalid/oversized worker frame')
        message = json.loads(wire)
        if message.get('type')!='rerank' or not isinstance(message.get('seq'),int):
            raise RuntimeError('Invalid protocol request')
        seq = message['seq']
        request_id = message['request']['request_id']
        try:
            with redirect_stdout(sys.stderr):
                request = Request.model_validate(message['request'])
                docs = sorted(request.documents,key=lambda d:d.document_id)
                ranked = backend.rerank(request.query,[d.text for d in docs],args.pair_batch_size)
                ranking = [{'document_id':docs[r['index']].document_id,'score':r['score']}
                           for r in ranked[:request.top_k]]
            send({'type':'result','seq':seq,'request_id':request_id,'ranking':ranking})
        except Exception as error:
            print(f'Inference error for {request_id}: {error}',file=sys.stderr,flush=True)
            send({'type':'result','seq':seq,'request_id':request_id,'error':'Inference failed'})


if __name__ == '__main__':
    main()
