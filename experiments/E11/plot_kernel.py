"""Offline plots of the measured fused-kernel operation and model benchmarks."""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE / 'kernel_benchmark.json').read_text())
    rows = result['model_benchmark']
    xs = [r['batch_size'] for r in rows]
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {'type': 'table'}]],
                        vertical_spacing=.2, horizontal_spacing=.12,
                        subplot_titles=['Whole model · prepared-input forward',
                                        'Whole model · complete scoring',
                                        'Residual + LayerNorm · isolated speedup',
                                        'Full validation · quality parity'])
    for col, scope in ((1, 'forward'), (2, 'scoring')):
        for mode, color in (('standard', '#2563eb'), ('fused', '#d97706')):
            values = [row[scope][mode]['pairs_per_second'] for row in rows]
            fig.add_trace(go.Scatter(x=xs, y=[v['q25'] for v in values],
                                    line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=col)
            fig.add_trace(go.Scatter(x=xs, y=[v['q75'] for v in values],
                                    line=dict(width=0), fill='tonexty', fillcolor=
                                    'rgba(37,99,235,.12)' if mode == 'standard' else 'rgba(217,119,6,.12)',
                                    showlegend=False, hoverinfo='skip'), row=1, col=col)
            fig.add_trace(go.Scatter(x=xs, y=[v['median'] for v in values], mode='lines+markers',
                                    name='Ordinary MLX' if mode == 'standard' else 'Fused Metal',
                                    legendgroup=mode, showlegend=(col == 1), line=dict(color=color, width=3)), row=1, col=col)
        fig.update_yaxes(title_text='Scored pairs / second', row=1, col=col)
    fig.add_trace(go.Bar(x=[str(x['batch_size']) for x in result['operation_microbench']],
                         y=[x['speedup'] for x in result['operation_microbench']],
                         marker_color='#0f766e', showlegend=False), row=2, col=1)
    fig.add_hline(y=1, line_dash='dash', row=2, col=1)
    fig.update_yaxes(title_text='Ordinary time / fused time', row=2, col=1)
    quality = result['full_validation_quality']
    fig.add_trace(go.Table(header=dict(values=['Metric', 'Ordinary MLX', 'Fused Metal']),
        cells=dict(values=[['Validation queries', 'nDCG@10', 'MRR@10', 'Rankings changed', 'Max score difference'],
                           [f'{quality["queries"]:,}', f'{quality["standard"]["ndcg_at_10"]:.6f}',
                            f'{quality["standard"]["mrr_at_10"]:.6f}', 'reference', 'reference'],
                           [f'{quality["queries"]:,}', f'{quality["fused"]["ndcg_at_10"]:.6f}',
                            f'{quality["fused"]["mrr_at_10"]:.6f}', quality['changed_rankings'],
                            f'{quality["max_score_absolute_error"]:.3g}']])), row=2, col=2)
    for col in (1, 2):
        fig.update_xaxes(type='log', tickmode='array', tickvals=xs,
                         ticktext=list(map(str, xs)), title_text='Pairs per batch', row=1, col=col)
    fig.update_xaxes(title_text='Pairs per batch', row=2, col=1)
    fig.update_layout(template='plotly_white', height=850, margin=dict(t=120, b=100, l=80, r=30),
                      title=dict(text='Epoch 3 · ordinary MLX vs fused residual + LayerNorm<br>'
                                      '<sup>FP32 · fixed validation inputs · 20 model repeats per point · shaded IQR · inference only</sup>', x=.04),
                      legend=dict(orientation='h', y=1.12),
                      annotations=list(fig.layout.annotations) + [dict(x=0, y=-.12, xref='paper', yref='paper',
                          showarrow=False, align='left', text=
                          'Whole-model scoring includes tokenization, input preparation and CPU-readable scores; no HTTP timing.<br>'
                          'Microbenchmarks alone do not establish application speedup. Ordinary MLX remains the default.')])
    path = HERE / 'kernel_interactive.html'
    fig.write_html(path, include_plotlyjs=True, config={'responsive': True, 'displaylogo': False})
    print(path.resolve())


if __name__ == '__main__': main()
