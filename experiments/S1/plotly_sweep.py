"""Render the saved benchmark sweeps as a self-contained interactive HTML file."""
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backend import HERE


def main():
    data = json.loads((HERE / 'sweep_results.json').read_text())
    fig = make_subplots(rows=2, cols=2, vertical_spacing=.18,
                        subplot_titles=['Batch size · scoring throughput',
                                        'Batch size · batch latency',
                                        'Concurrency · HTTP throughput',
                                        'Concurrency · request latency'])
    colors = {'pytorch': ('#2563eb', 'rgba(37,99,235,0.13)'),
              'mlx': ('#d97706', 'rgba(217,119,6,0.13)')}
    for row, kind, xkey in [(1,'batch','batch_size'),(2,'concurrency','concurrency')]:
        points = data[kind]
        xs = [p[xkey] for p in points]
        for backend in ('pytorch','mlx'):
            color, fill = colors[backend]
            name = 'PyTorch / MPS' if backend == 'pytorch' else 'MLX / Metal'
            values = [p[backend]['scoring'] if row==1 else p[backend] for p in points]
            keys = ('pairs_per_second','median_batch_ms') if row==1 else ('requests_per_second','median_ms')
            for col, key in enumerate(keys,1):
                series = [v[key] for v in values]
                fig.add_trace(go.Scatter(x=xs,y=[s['q25'] for s in series],mode='lines',
                    line=dict(width=0),showlegend=False,hoverinfo='skip',legendgroup=backend),row=row,col=col)
                fig.add_trace(go.Scatter(x=xs,y=[s['q75'] for s in series],mode='lines',
                    line=dict(width=0),fill='tonexty',fillcolor=fill,showlegend=False,
                    hoverinfo='skip',legendgroup=backend),row=row,col=col)
                unit = ('pairs/s' if row==1 else 'requests/s') if col==1 else 'ms'
                fig.add_trace(go.Scatter(x=xs,y=[s['median'] for s in series],mode='lines+markers',
                    name=name,legendgroup=backend,showlegend=(row==1 and col==1),
                    line=dict(color=color,width=3),marker=dict(size=8),
                    customdata=[[s['q25'],s['q75']] for s in series],
                    hovertemplate=f'{name}<br>Size: %{{x}}<br>Median: %{{y:.2f}} {unit}'
                    '<br>IQR: %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra></extra>'),row=row,col=col)
            if row == 2:
                fig.add_trace(go.Scatter(x=xs,y=[v['p95_ms']['median'] for v in values],mode='lines+markers',
                    name=f'{name} p95',legendgroup=backend,line=dict(color=color,dash='dash',width=2),
                    hovertemplate=name+'<br>Concurrency: %{x}<br>p95: %{y:.2f} ms<extra></extra>'),row=2,col=2)
        for col in (1,2):
            fig.update_xaxes(type='log',tickmode='array',tickvals=xs,ticktext=[str(x) for x in xs],
                title_text='Pairs per model batch' if row==1 else 'Concurrent HTTP requests',row=row,col=col)
    for row,col,title in [(1,1,'Scored pairs / second'),(1,2,'Median batch latency (ms)'),
                          (2,1,'Requests / second'),(2,2,'Request latency (ms)')]:
        fig.update_yaxes(title_text=title,rangemode='tozero',row=row,col=col)
    fig.update_layout(template='plotly_white',height=960,
        title=dict(text='Epoch 3 · PyTorch vs MLX<br><sup>Validation sample · Apple M5 GPU · FP32 · 6 repeats per point · shading = IQR</sup>',x=.05),
        margin=dict(t=110,b=100,l=85,r=45),hovermode='closest',
        legend=dict(orientation='h',y=1.09,x=0),
        annotations=list(fig.layout.annotations)+[dict(x=0,y=-.12,xref='paper',yref='paper',showarrow=False,
            align='left',text='HTTP: 2 documents/request · one worker · no cross-request batching. Test split untouched.<br>'
            'Shading shows repeat variability, not confidence intervals. Scroll/zoom and click legend entries to explore.')])
    output = HERE/'sweep_interactive.html'
    fig.write_html(output,include_plotlyjs=True,full_html=True,
                   config={'responsive':True,'scrollZoom':True,'displaylogo':False})
    print(output.resolve())


if __name__ == '__main__':
    main()
