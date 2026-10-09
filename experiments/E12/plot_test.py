"""Interactive test-set batch plots and full server quality table."""
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .test_compare import HERE


def main():
    result=json.loads((HERE/'test_comparison.json').read_text())
    points=result['direct_batch_sweep']
    xs=[r['pair_batch_size'] for r in points]
    fig=make_subplots(rows=2,cols=2,specs=[[{},{}],[{'type':'table','colspan':2},None]],
                      row_heights=[.65,.35],vertical_spacing=.18,
                      subplot_titles=['Direct inference · throughput','Direct inference · batch latency',
                                      'Full test split · HTTP evaluation (concurrency 4)'])
    colors={'pytorch':('#2563eb','rgba(37,99,235,.13)'),'mlx':('#d97706','rgba(217,119,6,.13)')}
    for name in ('pytorch','mlx'):
        color,fill=colors[name]
        label='PyTorch / MPS' if name=='pytorch' else 'MLX / Metal'
        for col,key in ((1,'pairs_per_second'),(2,'median_batch_ms')):
            values=[r[name][key] for r in points]
            fig.add_trace(go.Scatter(x=xs,y=[v['q25'] for v in values],line=dict(width=0),
                showlegend=False,hoverinfo='skip',legendgroup=name),row=1,col=col)
            fig.add_trace(go.Scatter(x=xs,y=[v['q75'] for v in values],line=dict(width=0),
                fill='tonexty',fillcolor=fill,showlegend=False,hoverinfo='skip',legendgroup=name),row=1,col=col)
            fig.add_trace(go.Scatter(x=xs,y=[v['median'] for v in values],mode='lines+markers',name=label,
                line=dict(color=color,width=3),legendgroup=name,showlegend=(col==1),
                hovertemplate=label+'<br>Batch: %{x}<br>Value: %{y:.2f}<extra></extra>'),row=1,col=col)
    for col in (1,2):
        fig.update_xaxes(type='log',tickmode='array',tickvals=xs,ticktext=list(map(str,xs)),title_text='Pairs per batch',row=1,col=col)
        fig.update_yaxes(title_text='Scored pairs / second' if col==1 else 'Median batch latency (ms)',rangemode='tozero',row=1,col=col)
    quality=result['full_test_quality']
    fig.add_trace(go.Table(header=dict(values=['HTTP server','GPU backend','Queries','nDCG@10','MRR@10','Median request ms','Rankings changed vs Python/PyTorch']),
        cells=dict(values=[[r['coordinator'] for r in quality],[r['backend'] for r in quality],
            [r['queries'] for r in quality],[f'{r["ndcg_at_10"]:.6f}' for r in quality],
            [f'{r["mrr_at_10"]:.6f}' for r in quality],[f'{r["median_request_ms"]:.2f}' for r in quality],
            [r['changed_rankings_vs_python_pytorch'] if r['baseline'] else 'reference' for r in quality]])),row=2,col=1)
    fig.update_layout(template='plotly_white',height=850,margin=dict(t=110,b=80,l=80,r=40),
        title=dict(text='Epoch 3 · fixed TEST evaluation<br><sup>Batch sweep: first 128 test queries · 6 repeats · shaded IQR · full quality: all 45,665 queries</sup>',x=.05),
        legend=dict(orientation='h',y=1.1),
        annotations=list(fig.layout.annotations)+[dict(x=0,y=-.10,xref='paper',yref='paper',showarrow=False,align='left',
            text='Direct batches combine pairs across queries. HTTP has mostly 2 documents/request; no cross-request batching.<br>'
                 'Constructed MS MARCO test lists; not a realistic retrieval benchmark. Model selected before test evaluation.')])
    path=HERE/'test_interactive.html'
    fig.write_html(path,include_plotlyjs=True,config={'responsive':True,'scrollZoom':True,'displaylogo':False})
    print(path.resolve())


if __name__=='__main__': main()
