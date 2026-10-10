"""Locate trained-checkpoint differences between CUDA inference dispatch paths."""
import json
from pathlib import Path
import torch
from .benchmark import load_fixture


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, rows, _, _ = load_fixture()
    flat = [(r['query'], c['text']) for r in rows for c in r['candidates']]
    inputs = tokenizer([q for q, _ in flat], [d for _, d in flat], padding=True,
        truncation='longest_first', max_length=128, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}
    report = {'tf32': torch.backends.cuda.matmul.allow_tf32, 'paths': {}, 'layers': []}
    with torch.inference_mode():
        outputs = {}
        for mode, fast in [('original', True), ('original', False), ('reference', False), ('triton', False)]:
            torch.backends.mha.set_fastpath_enabled(fast)
            name = f'{mode}_fast{fast}'
            outputs[name] = model(**inputs, kernel_mode=mode).cpu()
        reference = outputs['reference_fastFalse']
        for name, value in outputs.items():
            error = (value - reference).abs()
            changed = 0
            offset = 0
            for row in rows:
                n = len(row['candidates'])
                changed += int(value[offset:offset+n].argsort().tolist() != reference[offset:offset+n].argsort().tolist())
                offset += n
            report['paths'][name] = {'max_score_error': error.max().item(), 'mean_score_error': error.mean().item(),
                                    'changed_rankings': changed}
        ids = inputs['input_ids']
        positions = torch.arange(ids.shape[1], device='cuda')
        x = model.embedding_norm(model.token_embeddings(ids) + model.position_embeddings(positions)[None] +
                                  model.segment_embeddings(inputs['token_type_ids']))
        padding = inputs['attention_mask'] == 0
        for index, layer in enumerate(model.encoder_layers):
            torch.backends.mha.set_fastpath_enabled(True)
            fast = layer(x, src_key_padding_mask=padding)
            torch.backends.mha.set_fastpath_enabled(False)
            plain = layer(x, src_key_padding_mask=padding)
            report['layers'].append({'index': index, 'max_error_all': (fast-plain).abs().max().item(),
                                    'max_error_cls': (fast[:,0]-plain[:,0]).abs().max().item()})
            x = plain
    path = Path(__file__).with_name('cuda_dispatch_diagnostic.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
