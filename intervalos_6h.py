"""Resumen BA y AUROC media con IC95%; predicciones OOF guardadas, sin reentrenar.
Entrada: predicciones privadas de analizar.py --corte 6.
"""
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, f1_score

MODELOS = [
 ('referencia_mayoritaria','Referencia mayoritaria','Referencia mayoritaria'),
 ('solo_datos','Solo datos (logística)','Solo datos'),
 ('resnet_solo_imagen','ResNet18 · imagen sola','ResNet18'),
 ('efficientnet_solo_imagen','EfficientNet-B0 · imagen sola','EfficientNet-B0'),
 ('color_textura_solo_imagen','Color y textura · imagen sola','Color y textura'),
 ('svm_solo_imagen','EfficientNet + SVM · imagen sola','EfficientNet + SVM'),
 ('mnasnet_solo_imagen','MNASNet 1.0 · imagen sola','MNASNet 1.0'),
]
EXTRAS=[
 ('resnet_imagen_datos','ResNet18 + datos','ResNet18 + datos'),
 ('efficientnet_imagen_datos','EfficientNet-B0 + datos','EfficientNet-B0 + datos'),
 ('color_textura_imagen_datos','Color y textura + datos','Color y textura + datos'),
 ('svm_imagen_datos','EfficientNet + SVM + datos','EfficientNet + SVM + datos'),
 ('mnasnet_imagen_datos','MNASNet 1.0 + datos','MNASNet 1.0 + datos'),
]

def require(ok,msg):
 if not ok: raise ValueError(msg)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compatible(p):
 try:
  c=json.loads((p/'configuracion.json').read_text())
  return c.get('corte_horas')==6 and c.get('max_horas_inclusive')==24 and c.get('edad_minima')==18 and c.get('exclusion_padding') is False and c.get('cohorte')=={'imagenes':74,'personas':35} and (p/'referencia_mayoritaria_predicciones.csv').exists()
 except (OSError,ValueError): return False

def locate(root,explicit):
 if explicit:
  p=Path(explicit);require(compatible(p),'La carpeta no corresponde al análisis de 6 h, edad ≥18, sin filtro de padding, 35 personas/74 imágenes.');return p
 candidates=sorted(p.parent for p in (root/'resultados').glob('*/configuracion.json') if compatible(p.parent))
 require(len(candidates)==1,'Debe haber una sola ejecución compatible. Usa --resultados RUTA para elegirla. Encontradas:\n'+'\n'.join(map(str,candidates)))
 return candidates[0]

def pair_kernel(pos,neg):
 """AUROC entre pares de personas; cada persona pesa uno, imágenes 1/n."""
 return np.array([[np.mean((a[:,None]>b[None,:])+0.5*(a[:,None]==b[None,:])) for b in neg] for a in pos])

def auc_bootstrap(p,n=10000,seed=20260929):
 """Bootstrap por persona dentro de cada partición × clase.
 Conserva tamaños por estrato y calcula la media NO ponderada de 5 AUROC.
 Es condicional a las particiones/predicciones fijas; no reentrena.
 """
 rng=np.random.default_rng(seed);dist=np.zeros(n);points=[];details=[]
 for fold,g in p.groupby('grupo_evaluacion',sort=True):
  vectors=[]
  for c in [0,1]:
   vectors.append([r.puntuacion.to_numpy(float) for _,r in g[g.clase==c].groupby('persona_id',sort=True)])
  neg,pos=vectors;n0,n1=len(neg),len(pos)
  require(n0>0 and n1>0,f'Partición {fold} sin ambas clases; AUROC media no definida.')
  kernel=pair_kernel(pos,neg);point=kernel.mean()
  ref=roc_auc_score(g.clase,g.puntuacion,sample_weight=g.peso_evaluacion)
  require(np.isclose(point,ref,atol=1e-12),'Error en ponderación de AUROC.')
  # Misma semilla/orden de participantes en todos los modelos: remuestreo pareado.
  c0=rng.multinomial(n0,np.full(n0,1/n0),size=n)
  c1=rng.multinomial(n1,np.full(n1,1/n1),size=n)
  dist+=np.einsum('bi,ij,bj->b',c1,kernel,c0)/(n0*n1)
  points.append(point);details.append({'particion':int(fold),'personas_menos6h':n0,'personas_6a24h':n1,'AUROC':point})
 require(len(points)==5,'Se esperaban 5 particiones.')
 dist/=len(points)
 return float(np.mean(points)),np.quantile(dist,[.025,.975]),dist,details

def ba_ci(p,n=10000,seed=42):
 """IC95% percentil: bootstrap de personas por clase, 10000 réplicas por defecto."""
 x=p.assign(acierto=(p.clase==p.prediccion).astype(float)).groupby('persona_id',sort=True).agg(clase=('clase','first'),acierto=('acierto','mean'))
 rng=np.random.default_rng(seed);parts=[]
 for c in [0,1]:
  a=x.loc[x.clase==c,'acierto'].to_numpy();idx=rng.integers(0,len(a),size=(n,len(a)));parts.append(a[idx].mean(axis=1))
 return np.quantile((parts[0]+parts[1])/2,[.025,.975])

def validate(p,canonical,register):
 required=['persona_id','archivo','tiempo_horas','edad','clase','grupo_evaluacion','prediccion','puntuacion','peso_evaluacion']
 require(set(required)<=set(p.columns),'Faltan columnas en predicciones.')
 require(not p[required].isna().any().any(),'Hay valores faltantes en predicciones.')
 require(not p.duplicated(['persona_id','archivo']).any(),'Predicciones repetidas para la misma imagen.')
 require(len(p)==74 and p.persona_id.nunique()==35,'Cohorte diferente de 74 imágenes/35 personas.')
 require((p.edad>=18).all() and p.tiempo_horas.between(0,24).all(),'Edades o tiempos incompatibles.')
 require(np.array_equal(p.clase.to_numpy(),(p.tiempo_horas>=6).astype(int).to_numpy()),'Etiquetas incompatibles con corte 6 h.')
 require(p.prediccion.isin([0,1]).all() and np.isfinite(p.puntuacion).all(),'Predicciones o puntuaciones inválidas.')
 require((p.groupby('persona_id')[['clase','grupo_evaluacion','edad']].nunique()==1).all().all(),'Una persona aparece en más de una clase, edad o partición.')
 weights=1/p.groupby('persona_id').persona_id.transform('size')
 require(np.allclose(p.peso_evaluacion,weights),'Los pesos no dan igual peso a cada participante.')
 fields=['persona_id','archivo','tiempo_horas','edad','clase','grupo_evaluacion']
 if canonical is not None:
  require(p[fields].equals(canonical[fields]),'Modelos con participantes, imágenes, tiempos o particiones diferentes.')
 seen=set();allids=set(p.persona_id)
 require(len(register)==5,'Registro de particiones incompleto.')
 for item in register:
  train=set(item['personas_entrenamiento']);test=set(item['personas_evaluacion'])
  require(not train&test and train|test==allids,'Partición incompatible o con solapamiento.')
  require(not seen&test,'Una persona evaluada en dos particiones.');seen|=test
  require(set(p.loc[p.grupo_evaluacion==item['grupo'],'persona_id'])==test,'Predicciones no coinciden con particiones.json.')
 require(seen==allids,'No todas las personas tienen evaluación fuera de entrenamiento.')

METHOD='''ANÁLISIS PRINCIPAL: <6 frente a 6–24 horas, participantes ≥18 años, sin exclusión por padding.
Se resumen predicciones fuera de entrenamiento previamente guardadas. Cada persona pesa uno; sus imágenes se ponderan por 1/n. No se promedian fotografías para emitir una nueva predicción por persona.

AUROC media: media aritmética, no ponderada, de las AUROC de las cinco particiones. Clase positiva: 6–24 h. No es la AUROC obtenida al juntar puntuaciones de todas las particiones, ni un porcentaje de aciertos.
IC95% de AUROC media: percentiles 2,5 y 97,5 de 10000 remuestreos (por defecto) de personas, estratificados por partición de evaluación y clase. Cada persona remuestreada conserva todas sus imágenes y su multiplicidad. Las mismas réplicas se usan para todos los modelos. Se mantienen las cinco particiones; no se remuestrean solo cinco valores AUROC como si fueran cinco muestras independientes.
Los IC son aproximados y condicionales a las predicciones, a las particiones y a los tamaños por clase observados. No incluyen toda la variación del entrenamiento, la elección de modelos ni las múltiples pruebas. Los estratos con una sola persona no aportan variación de remuestreo dentro de ese estrato; pueden subestimarse las incertidumbres. No constituyen validación externa ni prueba de superioridad.

Exactitud balanceada: se conserva la estimación central del análisis previo y se actualiza el IC95% con 10000 remuestreos (por defecto) de personas estratificados por clase, semilla 42. Los límites pueden diferir ligeramente de los calculados anteriormente con 2000 repeticiones. Su esquema de remuestreo difiere del AUROC: este último se estratifica además por partición para mantener definidas las cinco AUROC.
Recall es sensibilidad por clase. Se muestran sensibilidad de <6 h y sensibilidad de 6–24 h; esta última es la sensibilidad para la clase positiva. F1 macro promedia F1 de ambas clases y no equivale a sensibilidad. La exactitud balanceada es la media de las dos sensibilidades.
Referencia mayoritaria: sus puntuaciones son constantes dentro de cada partición, por lo que su AUROC media e intervalo son 0,500. Ese intervalo degenerado es una propiedad del comparador, no una estimación precisa del rendimiento de un modelo clínico.
Solo datos es una referencia sin imagen. Las demás estrategias de la tabla principal usan imagen sola. --todos añade las variantes imagen + datos.
Documentación de métricas: https://scikit-learn.org/stable/modules/model_evaluation.html
'''

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--proyecto',default='datos_privados');ap.add_argument('--resultados');ap.add_argument('--salida');ap.add_argument('--bootstrap',type=int,default=10000,help='Número de réplicas para AMBOS IC: exactitud balanceada y AUROC (por defecto 10000)');ap.add_argument('--semilla',type=int,default=20260929);ap.add_argument('--todos',action='store_true');ap.add_argument('--sin-mostrar',action='store_true');args=ap.parse_args(argv)
 require(args.bootstrap>=2000,'Usa al menos 2000 réplicas.')
 root=Path(args.proyecto);src=locate(root,args.resultados)
 print('Predicciones fuente:',src,flush=True)
 print(f'Réplicas bootstrap: exactitud balanceada = {args.bootstrap}; AUROC media = {args.bootstrap}.',flush=True)
 source_table=pd.read_csv(src/'metricas.csv').set_index('modelo');register=json.loads((src/'particiones.json').read_text())
 specs=MODELOS+(EXTRAS if args.todos else []);rows=[];foldrows=[];distributions={};canonical=None;hashes={}
 for key,name,label in specs:
  path=src/f'{key}_predicciones.csv';p=pd.read_csv(path).sort_values(['persona_id','archivo']).reset_index(drop=True)
  validate(p,canonical,register)
  if canonical is None:canonical=p.copy()
  y=p.clase;pred=p.prediccion;w=p.peso_evaluacion
  ba=balanced_accuracy_score(y,pred,sample_weight=w);bci=ba_ci(p,args.bootstrap,42)
  sens=recall_score(y,pred,labels=[0,1],average=None,sample_weight=w,zero_division=0)
  f1=f1_score(y,pred,average='macro',sample_weight=w,zero_division=0)
  point,ci,boot,details=auc_bootstrap(p,args.bootstrap,args.semilla)
  old=source_table.loc[name]
  for col,value in [('exactitud_balanceada',ba),('ROC_AUC_media_grupos',point),('F1_macro',f1),('sensibilidad_menos6h',sens[0]),('sensibilidad_6a24h',sens[1])]:
   require(np.isclose(old[col],value,atol=1e-9),f'{name}: {col} no coincide con el resultado guardado. No se mezclan ejecuciones.')
  rows.append({'estrategia':label,'modelo_fuente':name,'exactitud_balanceada_pct':100*ba,'BA_IC95_inferior_pct':100*bci[0],'BA_IC95_superior_pct':100*bci[1],'sensibilidad_menos6h_pct':100*sens[0],'sensibilidad_6a24h_pct':100*sens[1],'F1_macro_pct':100*f1,'AUROC_media':point,'AUROC_media_IC95_inferior':ci[0],'AUROC_media_IC95_superior':ci[1]})
  foldrows.extend(dict(modelo=label,**d) for d in details);distributions[key]=boot;hashes[path.name]=sha(path)
  print(f'{label}: BA {100*ba:.2f}%; AUROC media {point:.3f} (IC95% {ci[0]:.3f}–{ci[1]:.3f})',flush=True)
 table=pd.DataFrame(rows);foldtable=pd.DataFrame(foldrows)
 stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC');dest=Path(args.salida) if args.salida else root/'resultados'/('resumen_IC_6h_'+stamp);dest.mkdir(parents=True,exist_ok=False)
 table.to_csv(dest/'resumen_metricas_IC95.csv',index=False,encoding='utf-8-sig');foldtable.to_csv(dest/'AUROC_por_particion.csv',index=False,encoding='utf-8-sig')
 nice=pd.DataFrame({'Estrategia':table.estrategia,'Exactitud balanceada %':table.exactitud_balanceada_pct.map(lambda x:f'{x:.2f}'),'IC95% exactitud balanceada':[f'{l:.2f}–{h:.2f}' for l,h in zip(table.BA_IC95_inferior_pct,table.BA_IC95_superior_pct)],'Sensibilidad <6 h %':table.sensibilidad_menos6h_pct.map(lambda x:f'{x:.2f}'),'Sensibilidad 6–24 h %':table.sensibilidad_6a24h_pct.map(lambda x:f'{x:.2f}'),'F1 macro %':table.F1_macro_pct.map(lambda x:f'{x:.2f}'),'AUROC media':table.AUROC_media.map(lambda x:f'{x:.3f}'),'IC95% AUROC media':[f'{l:.3f}–{h:.3f}' for l,h in zip(table.AUROC_media_IC95_inferior,table.AUROC_media_IC95_superior)]})
 nice.to_csv(dest/'tabla_para_informe.csv',index=False,encoding='utf-8-sig')
 (dest/'tabla_para_informe.html').write_text('<meta charset="utf-8"><style>body{font-family:Arial}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #bbb;text-align:left}</style><h2>Rendimiento para &lt;6 frente a 6–24 horas</h2>'+nice.to_html(index=False,border=0)+'<p>35 personas; 74 imágenes. Igual peso por persona. IC95% aproximados sobre predicciones fijas; AUROC media de cinco particiones.</p>',encoding='utf-8')
 np.savez_compressed(dest/'replicas_AUROC_media.npz',**distributions)
 method=METHOD.replace('10000 remuestreos (por defecto)',f'{args.bootstrap} remuestreos')
 sparse=foldtable[(foldtable.personas_menos6h==1)|(foldtable.personas_6a24h==1)][['particion','personas_menos6h','personas_6a24h']].drop_duplicates()
 if len(sparse):
  msg='AVISO: estratos con una sola persona:\n'+sparse.to_string(index=False);print(msg);method+='\n'+msg+'\n'
 (dest/'METODO_Y_LIMITACIONES.txt').write_text(method,encoding='utf-8')
 for f in ['configuracion.json','particiones.json','metricas.csv']:hashes[f]=sha(src/f)
 (dest/'trazabilidad.json').write_text(json.dumps({'fuente':str(src),'sha256':hashes,'version_resumen':'2026-09-30_BA_AUROC_10000','bootstrap_BA':args.bootstrap,'semilla_BA':42,'bootstrap_AUROC':args.bootstrap,'semilla_AUROC':args.semilla,'cohorte':{'personas':35,'imagenes':74},'AUROC':'media no ponderada entre particiones','IC_AUROC':'percentil, personas estratificadas por particion y clase, predicciones fijas','IC_BA':f'{args.bootstrap} bootstrap, semilla 42, personas estratificadas por clase','versiones':{'numpy':np.__version__,'pandas':pd.__version__}},ensure_ascii=False,indent=2))
 h=4.9 if len(table)<=7 else 7.0
 fig,axes=plt.subplots(1,2,figsize=(13,h),sharey=True);yy=np.arange(len(table));color='#246780'
 for ax,pointcol,locol,hicol,title,xmax,ticks in [(axes[0],'exactitud_balanceada_pct','BA_IC95_inferior_pct','BA_IC95_superior_pct','Exactitud balanceada (%)',100,np.arange(0,101,20)),(axes[1],'AUROC_media','AUROC_media_IC95_inferior','AUROC_media_IC95_superior','AUROC media',1,np.arange(0,1.01,.2))]:
  ax.hlines(yy,table[locol],table[hicol],color=color,lw=1.6);ax.scatter(table[pointcol],yy,color=color,s=38,zorder=3)
  ax.axvline(xmax/2,color='#888',ls='--',lw=1);ax.set_xlim(0,xmax);ax.set_xticks(ticks);ax.set_xlabel(title,fontsize=11);ax.set_yticks(yy);ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
  for i,r in table.iterrows():
   fmt='.2f' if xmax==100 else '.3f';label=f'{r[pointcol]:{fmt}} [{r[locol]:{fmt}}–{r[hicol]:{fmt}}]'
   ax.annotate(label,(r[pointcol],i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8,color='#333')
 axes[0].set_yticklabels(table.estrategia,fontsize=10);axes[0].invert_yaxis();axes[0].set_ylim(len(table)-.55,-.65)
 fig.suptitle('Clasificación de equimosis: <6 frente a 6–24 horas',fontsize=14,y=.97)
 fig.text(.5,.025,'35 personas · 74 imágenes · IC95% aproximados sobre predicciones fijas\nAUROC: media de cinco particiones; bootstrap por persona dentro de partición y clase.',ha='center',fontsize=9)
 fig.subplots_adjust(left=.21,right=.98,bottom=.20,top=.86,wspace=.22)
 for ext in ['png','pdf','svg']:fig.savefig(dest/f'resumen_BA_AUROC_IC95.{ext}',dpi=300,bbox_inches='tight')
 print('\nResultados guardados en:',dest,flush=True)
 try:
  from IPython.display import display
  display(nice)
 except ImportError: print(nice.to_string(index=False))
 if not args.sin_mostrar:plt.show()
 plt.close(fig)
 return dest

if __name__=='__main__': main()
