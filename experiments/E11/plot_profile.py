"""Render synchronized inference profiling measurements as an offline Plotly report."""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent


def main():
    report = json.loads((HERE / 'profiling.json').read_text())
    rows = report['results']
    sizes = [row['batch_size'] for row in rows]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=.2, horizontal_spacing=.15,
                        subplot_titles=['Normal scoring vs prepared-input forward',
                                        'Instrumented scoring phases',
                                        'Batch 2 · isolated operation categories',
                                        'Batch 128 · isolated operation categories'])
    for name, label, color in [('normal_scoring', 'Normal scoring', '#2563eb'),
                               ('resident_forward', 'Prepared-input forward', '#d97706')]:
        values = [row['phases'][name] for row in rows]
        fig.add_trace(go.Scatter(x=sizes, y=[v['median_ms'] for v in values],
                                error_y=dict(type='data', symmetric=False,
                                    array=[v['q75_ms'] - v['median_ms'] for v in values],
                                    arrayminus=[v['median_ms'] - v['q25_ms'] for v in values]),
                                name=label, mode='lines+markers', line=dict(color=color)), row=1, col=1)
    for name in ('tokenization', 'input_preparation', 'model_forward', 'output_conversion'):
        fig.add_trace(go.Bar(x=sizes, y=[row['phases'][name]['median_ms'] for row in rows],
                             name=name.replace('_', ' ').title()), row=1, col=2)
    for col, size in ((1, 2), (2, 128)):
        row = next(r for r in rows if r['batch_size'] == size)
        categories = list(reversed(row['categories']))
        fig.add_trace(go.Bar(x=[c['isolated_sum_ms'] for c in categories],
                             y=[c['category'].replace('_', ' ') for c in categories],
                             orientation='h', showlegend=False, marker_color='#0f766e',
                             hovertemplate='%{y}<br>Isolated sum: %{x:.3f} ms<extra></extra>'), row=2, col=col)
        fig.update_xaxes(title_text='Sum of isolated operation medians (ms)', row=2, col=col)
    for col in (1, 2):
        fig.update_xaxes(type='log', tickmode='array', tickvals=sizes,
                         ticktext=list(map(str, sizes)), title_text='Pairs per batch', row=1, col=col)
        fig.update_yaxes(title_text='Milliseconds', row=1, col=col)
    fig.update_layout(template='plotly_white', barmode='stack', height=950,
                      margin=dict(t=130, b=130, l=150, r=40),
                      title=dict(text='Epoch 3 · MLX inference profile<br><sup>FP32 · fixed validation inputs · 30 repeats · synchronized wall times · no HTTP</sup>', x=.04),
                      legend=dict(orientation='h', y=1.11),
                      annotations=list(fig.layout.annotations) + [dict(x=0, y=-.15, xref='paper', yref='paper',
                          showarrow=False, align='left', text=
                          'Operation sums include dispatch and synchronization for every isolated operation; they are not whole-model GPU-time shares.<br>'
                          'Instrumented phases insert a preparation barrier. Normal scoring is measured separately. Metal traces are saved for Xcode inspection.')])
    path = HERE / 'profiling_interactive.html'
    fig.write_html(path, include_plotlyjs=True, config={'responsive': True, 'displaylogo': False})
    print(path.resolve())


if __name__ == '__main__':
    main()
