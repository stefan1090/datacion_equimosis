"""Funciones originales del análisis de relleno, aplicadas a la cohorte adulta."""
from pathlib import Path

from datetime import datetime, timezone

import argparse, hashlib, json, shutil, unicodedata, warnings, importlib.metadata

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.exceptions import ConvergenceWarning

VERSION='2026-09-28_padding50_v1'

MODELOS={'efficientnet':'EfficientNet-B0','mnasnet':'MNASNet 1.0'}

UMBRAL=50.0

NBOOT=2000

SEED=42

SCENARIOS={'A':'A · Sin exclusión','B':'B · Mismo modelo, imágenes conservadas','C':'C · Reentrenado, imágenes conservadas'}

def check(condition,message):
    if not condition: raise ValueError(message)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def weights(ids):
    counts=pd.Series(ids).value_counts()
    return np.array([1/counts[x] for x in ids],float)

def fit_predict(X,y,ids,train,test):
    check(len(train)>0 and len(test)>0,'Partición vacía después de excluir: no se pueden conservar las cinco particiones.')
    check(set(y[train])=={0,1},'Falta una clase en entrenamiento después de excluir.')
    check(not set(ids[train])&set(ids[test]),'Una persona aparece en entrenamiento y evaluación.')
    wp=weights(ids[train])
    people=pd.DataFrame({'id':ids[train],'y':y[train]}).drop_duplicates('id')
    counts=people.y.value_counts()
    wc=wp*np.array([len(people)/(2*counts[c]) for c in y[train]])
    scaler=StandardScaler().fit(X[train],sample_weight=wp/wp.mean())
    clf=LogisticRegression(C=.01,solver='lbfgs',max_iter=5000,random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        clf.fit(scaler.transform(X[train]),y[train],sample_weight=wc/wc.mean())
    Z=scaler.transform(X[test])
    return clf.predict(Z),clf.predict_proba(Z)[:,1]

def person_accuracy(p):
    q=p.assign(acierto=(p.clase==p.prediccion).astype(float))
    return q.groupby('persona_id',sort=True).agg(clase=('clase','first'),acierto=('acierto','mean'))

def boot_ba(p):
    q=person_accuracy(p);rng=np.random.default_rng(SEED);means=[]
    for c in [0,1]:
        z=q.loc[q.clase==c,'acierto'].to_numpy()
        check(len(z)>0,'Se necesita al menos una persona de cada clase.')
        means.append(z[rng.integers(0,len(z),size=(NBOOT,len(z)))].mean(axis=1))
    return (means[0]+means[1])/2

def metrics(p,model,scenario):
    w=weights(p.persona_id.to_numpy());cm=confusion_matrix(p.clase,p.prediccion,labels=[0,1],sample_weight=w,normalize='true')
    areas=[]
    for _,z in p.groupby('grupo_evaluacion'):
        check(set(z.clase)=={0,1},'Falta una clase en evaluación. No se puede calcular AUROC media en las mismas cinco particiones.')
        areas.append(roc_auc_score(z.clase,z.puntuacion,sample_weight=weights(z.persona_id.to_numpy())))
    dist=boot_ba(p);lo,hi=np.quantile(dist,[.025,.975])
    return {'modelo':model,'escenario':scenario,'descripcion':SCENARIOS[scenario],'imagenes':len(p),'personas':p.persona_id.nunique(),
      'exactitud':accuracy_score(p.clase,p.prediccion,sample_weight=w),
      'exactitud_balanceada':balanced_accuracy_score(p.clase,p.prediccion,sample_weight=w),
      'BA_IC95_inferior':lo,'BA_IC95_superior':hi,'sensibilidad_menos6h':cm[0,0],'sensibilidad_6a24h':cm[1,1],
      'F1_macro':f1_score(p.clase,p.prediccion,sample_weight=w,average='macro',zero_division=0),
      'AUROC_media_particiones':float(np.mean(areas)),'AUROC_agrupada':roc_auc_score(p.clase,p.puntuacion,sample_weight=w)},dist

"""Funciones originales del análisis de relleno, aplicadas a la cohorte adulta."""
from pathlib import Path

from datetime import datetime, timezone

import argparse

import hashlib

import json

import shutil

import unicodedata

import importlib.metadata

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

LABELS = ['Menos de 6 h', '6–24 h']

COLORS = ['#26738c', '#d87732']

def check(ok, message):
    if not ok:
        raise ValueError(message)

def wquantile(x, w, probabilities):
    order = np.argsort(x)
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    return np.interp(probabilities, (np.cumsum(w)-w/2)/w.sum(), x)

def analyse(a, nboot=5000, seed=42):
    # Matriz de concordancia: una celda por pareja de personas de clases opuestas.
    # Promediar sobre sus lesiones equivale exactamente al AUROC ponderado por persona.
    samples = [[z.padding_porcentaje.to_numpy() for _,z in a[a.clase==c].groupby('persona_id',sort=True)] for c in [0,1]]
    n0,n1 = map(len,samples)
    check(min(n0,n1)>=2, 'Se necesitan al menos dos personas por clase para los intervalos.')
    kernel = np.array([[np.mean((p[:,None]>n[None,:]) + .5*(p[:,None]==n[None,:])) for n in samples[0]] for p in samples[1]])
    auc = roc_auc_score(a.clase,a.padding_porcentaje,sample_weight=a.peso_persona)
    check(np.isclose(auc,kernel.mean()), 'Fallo de comprobación del AUROC por persona.')
    rng = np.random.default_rng(seed)
    b0 = rng.multinomial(n0,np.full(n0,1/n0),size=nboot)/n0
    b1 = rng.multinomial(n1,np.full(n1,1/n1),size=nboot)/n1
    boot_auc = np.einsum('bi,ij,bj->b',b1,kernel,b0)
    means0 = np.array([x.mean() for x in samples[0]])
    means1 = np.array([x.mean() for x in samples[1]])
    boot_diff = b1@means1-b0@means0
    lo,hi = np.quantile(boot_auc,[.025,.975])
    dl,dh = np.quantile(boot_diff,[.025,.975])
    metrics = pd.DataFrame([
        dict(medida='AUROC: mayor padding indica 6–24 h',estimacion=auc,IC95_inferior=lo,IC95_superior=hi),
        dict(medida='AUROC inversa: menor padding indica 6–24 h',estimacion=1-auc,IC95_inferior=1-hi,IC95_superior=1-lo),
        dict(medida='Diferencia media de padding: 6–24 h menos <6 h (puntos porcentuales)',estimacion=means1.mean()-means0.mean(),IC95_inferior=dl,IC95_superior=dh)])
    rows=[]
    for c in [0,1]:
        z=a[a.clase==c];x=z.padding_porcentaje.to_numpy();w=z.peso_persona.to_numpy()
        q=wquantile(x,w,[.25,.5,.75])
        rows.append(dict(clase=LABELS[c],personas=z.persona_id.nunique(),imagenes=len(z),
            media_padding_pct=np.average(x,weights=w),Q1_padding_pct=q[0],mediana_padding_pct=q[1],Q3_padding_pct=q[2],
            minimo_pct=x.min(),maximo_pct=x.max(),imagenes_padding_mayor50=int((x>50).sum()),
            porcentaje_mayor50_ponderado=100*np.average(x>50,weights=w)))
    return metrics,pd.DataFrame(rows),pd.DataFrame({'AUROC':boot_auc,'diferencia_media_padding_pp':boot_diff})

from types import SimpleNamespace
mp=SimpleNamespace(MODELOS=MODELOS,weights=weights,fit_predict=fit_predict,metrics=metrics)
ma=SimpleNamespace(analyse=analyse)
"""Sensibilidad al relleno >50% y asociación relleno/clase en la cohorte principal adulta."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from sklearn.metrics import roc_curve
from metodos import base,cargar,particiones

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--proyecto',type=Path,required=True)
 p.add_argument('--caracteristicas',type=Path,required=True)
 p.add_argument('--principal',type=Path,required=True)
 p.add_argument('--salida',type=Path,required=True)
 args=p.parse_args();OUT=args.salida;S6=args.principal
 cfg=json.loads((S6/'configuracion.json').read_text())
 if cfg['corte_horas']!=6 or cfg['edad_minima']!=18:raise ValueError('Se necesita el principal de 6 h y edad ≥18.')
 d,excluded=base(args.proyecto,6)
 cache,hashes=cargar(args.proyecto,args.caracteristicas,d,mp.MODELOS)
 parts=particiones(S6/'particiones.json',d)
 fold=np.zeros(len(d),int)
 for f,(_,te) in enumerate(parts,1):fold[te]=f
 OUT.mkdir(parents=True,exist_ok=False)
 rows=[]
 for _,r in d.iterrows():
  with Image.open(args.proyecto/'imagenes'/r.archivo) as im:
   im=ImageOps.exif_transpose(im);w,h=im.size
  cw,ch=(224,round(h/w*224)) if w>=h else (round(w/h*224),224)
  pct=100*(1-cw*ch/224**2)
  rows.append(dict(persona_id=r.persona_id,archivo=r.archivo,clase=r.clase,relleno_porcentaje=pct,excluir=pct>50))
 aud=pd.DataFrame(rows);keep=~aud.excluir.to_numpy()
 aud.to_csv(OUT/'auditoria_privada.csv',index=False)
 pa=aud.rename(columns={'relleno_porcentaje':'padding_porcentaje'}).copy()
 pa['peso_persona']=mp.weights(pa.persona_id.to_numpy())
 metrics,desc,boot=ma.analyse(pa)
 metrics.to_csv(OUT/'padding_clase_metricas.csv',index=False)
 desc.to_csv(OUT/'padding_clase_descriptivos.csv',index=False)
 def save(fig,name):
  for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight')
  plt.close(fig)

 # A: principal verificado; B: mismas predicciones, casos conservados; C: reentrenado.
 padrows=[];pairs=[];preds={}
 for key,title in mp.MODELOS.items():
  A=pd.read_csv(S6/f'{key}_solo_imagen_predicciones.csv').set_index('archivo').loc[d.archivo].reset_index()
  assert np.array_equal(A.grupo_evaluacion,fold) and np.array_equal(A.persona_id,d.persona_id)
  assert np.array_equal(A.clase,d.clase) and np.allclose(A.tiempo_horas,d.tiempo_horas)
  X=cache[key];ids=d.persona_id.to_numpy();y=d.clase.to_numpy()
  pr=np.full(len(d),-1);sc=np.full(len(d),np.nan)
  for tr,te in parts:
   a=tr[keep[tr]];b=te[keep[te]]
   assert set(y[b])=={0,1}
   pr[b],sc[b]=mp.fit_predict(X,y,ids,a,b)
  B=A.loc[keep].copy();C=B.copy();C['prediccion']=pr[keep];C['puntuacion']=sc[keep]
  distributions={};results={}
  for scenario,p in [('A',A),('B',B),('C',C)]:
   p['peso_evaluacion']=mp.weights(p.persona_id.to_numpy())
   row,dist=mp.metrics(p,title,scenario);padrows.append(row);distributions[scenario]=dist;results[scenario]=row
   p.to_csv(OUT/f'{key}_{scenario}_predicciones.csv',index=False);preds[key,scenario]=p
  delta=distributions['C']-distributions['B'];lo,hi=np.quantile(delta,[.025,.975])
  pairs.append(dict(modelo=title,diferencia_BA_pp=100*(results['C']['exactitud_balanceada']-results['B']['exactitud_balanceada']),IC95_inferior_pp=lo*100,IC95_superior_pp=hi*100,diferencia_AUROC_media=results['C']['AUROC_media_particiones']-results['B']['AUROC_media_particiones']))
 pad=pd.DataFrame(padrows);pair=pd.DataFrame(pairs)
 pad.to_csv(OUT/'metricas_ABC.csv',index=False);pair.to_csv(OUT/'diferencias_C_B.csv',index=False)
 
 # Padding: asociación.
 fig,axes=plt.subplots(1,2,figsize=(7,3.2))
 for c,color in [(0,'#26738c'),(1,'#d87732')]:
  z=pa[pa.clase==c].sort_values('padding_porcentaje');w=z.peso_persona.to_numpy();axes[0].step(np.r_[0,z.padding_porcentaje,100],np.r_[0,100*np.cumsum(w)/w.sum(),100],where='post',label=['<6 h','6–24 h'][c],color=color)
 axes[0].set(xlabel='Relleno (%)',ylabel='Porcentaje acumulado',xlim=(0,100),ylim=(0,102));axes[0].legend()
 x,y,_=roc_curve(pa.clase,pa.padding_porcentaje,sample_weight=pa.peso_persona);r=metrics.iloc[0]
 axes[1].plot(x,y,label=f'AUROC {r.estimacion:.3f}\nIC95% {r.IC95_inferior:.3f}–{r.IC95_superior:.3f}');axes[1].plot([0,1],[0,1],':',color='gray');axes[1].set(xlabel='1 − especificidad',ylabel='Sensibilidad 6–24 h',xlim=(0,1),ylim=(0,1));axes[1].legend(loc='lower right',fontsize=8);fig.tight_layout();save(fig,'padding_asociacion')
 fig,axes=plt.subplots(1,2,figsize=(7,3.3))
 for ax,(key,title) in zip(axes,mp.MODELOS.items()):
  r=pad[pad.modelo==title];x=np.arange(3);v=r.exactitud_balanceada.to_numpy()*100
  ax.hlines([],[],[]);ax.errorbar(x,v,yerr=np.vstack([v-r.BA_IC95_inferior.to_numpy()*100,r.BA_IC95_superior.to_numpy()*100-v]),fmt='o',capsize=4,color='#26738c');ax.axhline(50,color='gray',ls='--');ax.set(xticks=x,xticklabels=['A','B','C'],ylim=(0,100),title=title,ylabel='Exactitud balanceada (%)')
 fig.tight_layout();save(fig,'padding_ABC')
 fig,axes=plt.subplots(1,2,figsize=(7,3.3));grid=np.linspace(0,1,1001)
 for ax,(key,title) in zip(axes,mp.MODELOS.items()):
  for scenario in ['B','C']:
   p=preds[key,scenario];curves=[]
   for _,z in p.groupby('grupo_evaluacion'):
    xx,yy,_=roc_curve(z.clase,z.puntuacion,sample_weight=z.peso_evaluacion);curves.append(np.interp(grid,xx,yy))
   area=pad[(pad.modelo==title)&(pad.escenario==scenario)].AUROC_media_particiones.iloc[0]
   ax.plot(grid,np.mean(curves,axis=0),label=f'{scenario}: {area:.3f}')
  ax.plot([0,1],[0,1],':',color='gray');ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 − especificidad',ylabel='Sensibilidad 6–24 h',title=title);ax.legend(loc='lower right')
 fig.tight_layout();save(fig,'padding_ROC_BC')

 print('Resultados privados de padding:',OUT)
if __name__=='__main__':main()
