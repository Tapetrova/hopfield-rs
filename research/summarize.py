"""Aggregate by memory-set seed, not by individual images as independent units."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'results/run-2026-09-26'
def ci(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260926)
    boot = rng.choice(a, size=(10000, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), *np.quantile(boot,[.025,.975]).tolist()

parts=[]
for dataset,folder in [('MNIST exploratory','run-2026-09-26'),('MNIST confirmation','confirm-2026-09-26'),('Fashion-MNIST','fashion-2026-09-26')]:
    f=ROOT/'results'/folder/'mnist.csv'
    if not f.exists(): continue
    m=pd.read_csv(f)
    m['success']=m.overlap>=.95
    per=m.groupby(['seed','k','mask','rule'])[['success','exact_hidden','hidden_accuracy','converged']].mean().reset_index()
    per.insert(0,'dataset',dataset);parts.append(per)
all_seeds=pd.concat(parts,ignore_index=True)
all_seeds.to_csv(ROOT/'seed_summary.csv',index=False,float_format='%.6f')
nearest=[]
for dataset,folder in [('MNIST','confirm-2026-09-26'),('Fashion-MNIST','fashion-2026-09-26')]:
    d=pd.read_csv(ROOT/'results'/folder/'nearest.csv')
    d['success']=d.overlap>=.95
    g=d.groupby(['seed','k','mask'])[['success','exact_hidden','hidden_accuracy','cue_matches']].mean().reset_index()
    g.insert(0,'dataset',dataset);nearest.append(g)
pd.concat(nearest,ignore_index=True).to_csv(ROOT/'nearest_seed_summary.csv',index=False,float_format='%.6f')
summary=[]
for keys,group in all_seeds.groupby(['dataset','k','mask','rule']):
    avg,low,high=ci(group.success)
    summary.append(dict(zip(['dataset','k','mask','rule'],keys))|dict(mean=avg,low=low,high=high,
        exact=group.exact_hidden.mean(),hidden_accuracy=group.hidden_accuracy.mean(),sets=len(group),converged=group.converged.mean()))
summary=pd.DataFrame(summary);summary.to_csv(ROOT/'retrieval_summary.csv',index=False,float_format='%.6f')
effects=[]
for keys,g in all_seeds[all_seeds.dataset!='MNIST exploratory'].groupby(['dataset','k','mask']):
    wide=g.pivot(index='seed',columns='rule',values='success')
    avg,lo,hi=ci(wide.projection_zero-wide.projection_diag)
    effects.append(dict(zip(['dataset','k','mask'],keys))|dict(difference=avg,low=lo,high=hi))
pd.DataFrame(effects).to_csv(ROOT/'paired_effects.csv',index=False,float_format='%.6f')
c=pd.read_csv(RUN/'capacity.csv');c['one_success']=c.one_overlap>=.95;c['final_success']=c.final_overlap>=.95
c.groupby(['n','p','noise','alpha'])[['one_success','final_success','converged','sweeps']].mean().to_csv(ROOT/'capacity_summary.csv',float_format='%.6f')
b=pd.read_csv(RUN/'benchmark.csv')
paired=b.pivot(index=['n','mode','seed'],columns='engine',values='seconds');paired['speedup']=paired.dense/paired.packed
paired.groupby(['n','mode']).median().to_csv(ROOT/'benchmark_summary.csv',float_format='%.6f')

fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
curves=c[(c.n==4096)&(c.noise==.05)].groupby('alpha')[['one_success','final_success']].mean()
axes[0].plot(curves.index,curves.one_success,'o-',label='One sweep')
axes[0].plot(curves.index,curves.final_success,'s-',label='Converged')
axes[0].set(title='Random patterns: N=4096, 50 sets',xlabel='Load P/N',ylabel='Retrieval success (overlap >= 0.95)',ylim=(-.03,1.03))
axes[0].legend();axes[0].grid(alpha=.2)
s=summary[(summary.dataset=='MNIST confirmation')&(summary.k==100)]
for mask,offset,label in [('lower_half',-.18,'Missing lower half'),('random_half',.18,'Missing random half')]:
    v=s[s['mask']==mask].set_index('rule').loc[['projection_diag','projection_half','projection_zero']]
    axes[1].bar(np.arange(3)+offset,v['mean'],width=.34,label=label)
    axes[1].errorbar(np.arange(3)+offset,v['mean'],yerr=[v['mean']-v.low,v.high-v['mean']],fmt='none',ecolor='black',capsize=3)
axes[1].set(title='MNIST: 100 memories, 20 new sets',ylabel='Retrieval success',ylim=(0,1.06),xticks=np.arange(3),xticklabels=['Full diagonal','Half diagonal','Zero diagonal'])
axes[1].legend(loc='upper left');axes[1].grid(axis='y',alpha=.2)
fig.savefig(ROOT/'results_overview.png',dpi=180)
fig.savefig(ROOT/'results_overview.svg')
files=list((ROOT/'results').glob('*/**/*.csv'))+[ROOT.parent/'hopfield/src/bin/research.rs',ROOT.parent/'hopfield/tests/research_audit.rs']
manifest={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
          'platform':platform.platform(),'upstream':'908fd3d24cd92ec571ecb44383731e5948571a58',
          'rust':'1.98.1 (48a229cea 2026-09-01)','bootstrap':'10000 resamples over memory-set seeds; paired differences',
          'sha256':{str(f.relative_to(ROOT.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(summary.to_string(index=False))
print(pd.DataFrame(effects).to_string(index=False))
