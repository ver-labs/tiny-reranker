"""Keep per-query diagnostics auditable without duplicating them in Git summaries."""
import gzip
import json
from .common import HERE,save,sha


def main():
    for path in sorted(HERE.glob('*_accuracy.json'))+sorted(HERE.glob('*_reserve.json')):
        report=json.loads(path.read_text());target=report.get('accuracy',report)
        if 'per_query' not in target:continue
        artifact=HERE/'scores'/f'{path.stem}_deltas.json.gz'
        artifact.parent.mkdir(exist_ok=True)
        with gzip.open(artifact,'wt') as stream:json.dump(target['per_query'],stream,allow_nan=False)
        with gzip.open(artifact,'rt') as stream:assert json.load(stream)==target['per_query']
        target.pop('per_query')
        target['per_query_artifact']=str(artifact.relative_to(HERE))
        target['per_query_artifact_sha256']=sha(artifact)
        save(path,report)


if __name__=='__main__':main()
