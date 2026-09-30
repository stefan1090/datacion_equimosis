"""Análisis principal (6 h) y sensibilidad (12 h), adultos ≥18 años y tiempos ≤24 h."""
import argparse
from pathlib import Path
import json
import importlib
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from metodos import base,cargar,particiones,sha

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proyecto',type=Path,required=True)
    p.add_argument('--caracteristicas',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--corte',type=int,choices=[6,12],required=True)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--particiones',type=Path,help='Registro privado de asignación por persona.')
    g.add_argument('--crear-particiones',action='store_true',help='Nueva asignación: no reproduce las particiones históricas de 6 h.')
    p.add_argument('--modelos',nargs='+',choices=['resnet','efficientnet','color_textura','svm','mnasnet'],default=['resnet','efficientnet','color_textura','svm','mnasnet'])
    a=p.parse_args(argv)
    if a.corte==6 and not a.particiones and not a.crear_particiones:
        p.error('Para reproducir 6 h se requiere --particiones; para una ejecución nueva, --crear-particiones.')
    m=importlib.import_module('metodos')
    m.configurar(a.corte)
    m.FAMILIAS={k:v for k,v in m.FAMILIAS.items() if k in a.modelos}
    d,ex=base(a.proyecto,a.corte);cache,huellas=cargar(a.proyecto,a.caracteristicas,d,m.FAMILIAS)
    ids=d.persona_id.to_numpy();y=d.clase.to_numpy();meta=d[m.META]
    folds=particiones(a.particiones,d) if a.particiones else m.particiones_persona(ids,y)
    if 'svm' in a.modelos:
        for f,(tr,te) in enumerate(folds,1):m.particiones_persona(ids[tr],y[tr],3,m.SEMILLA+f)
    a.salida.mkdir(parents=True,exist_ok=False);out=a.salida
    d.to_csv(out/'datos_usados.csv',index=False);ex.to_csv(out/'excluidos.csv',index=False);huellas.to_csv(out/'huellas_imagenes.csv',index=False)
    fs=np.zeros(len(d),int);registro=[]
    for f,(tr,te) in enumerate(folds,1):
        fs[te]=f
        registro.append(dict(grupo=f,personas_entrenamiento=sorted(set(ids[tr])),personas_evaluacion=sorted(set(ids[te]))))
    (out/'particiones.json').write_text(json.dumps(registro,indent=2))
    d.groupby('clase').agg(personas=('persona_id','nunique'),imagenes=('archivo','count')).to_csv(out/'distribucion_clases.csv')
    people=d.drop_duplicates('persona_id').sort_values('persona_id')
    def replicas(n):
        rng=np.random.default_rng(42)
        return {c:rng.integers(0,int((people.clase==c).sum()),size=(n,int((people.clase==c).sum()))) for c in [0,1]}
    # Principal actualizado a 10000. Aporte de datos y sensibilidad conservan 2000.
    nboot=10000 if a.corte==6 else 2000
    boot=replicas(nboot);boot_pares=replicas(2000)
    rows=[];dist={};res={}
    bp=np.zeros(len(d),int);bs=np.zeros(len(d))
    for tr,te in folds:
        prevalence=d.iloc[tr].drop_duplicates('persona_id').clase.mean()
        bp[te]=int(prevalence>.5);bs[te]=prevalence
    row,_=m.metricas('Referencia mayoritaria',y,bp,bs,ids,fs,boot);rows.append(row)
    ref=d.assign(grupo_evaluacion=fs,prediccion=bp,puntuacion=bs,peso_evaluacion=m.pesos_persona(ids))
    ref.to_csv(out/'referencia_mayoritaria_predicciones.csv',index=False)
    specs=[('solo_datos','Solo datos (logística)','datos',False,True,False)]
    for family,title in m.FAMILIAS.items():
        for datos in [False,True]:
            specs.append((family+('_imagen_datos' if datos else '_solo_imagen'),title+(' + datos' if datos else ' · imagen sola'),family,True,datos,family=='svm'))
    for key,title,fam,imagen,datos,svm in specs:
        print('Evaluando:',title,flush=True)
        X=cache['efficientnet' if svm else fam] if imagen else np.empty((len(d),0))
        pr,sc,ff=m.evaluar_variante(key,fam,X,meta,d,folds,imagen,datos,svm,out)
        row,_=m.metricas(title,y,pr,sc,ids,ff,boot);rows.append(row);res[key]=row
        _,dist[key]=m.metricas(title,y,pr,sc,ids,ff,boot_pares)
    pairs=[]
    for fam,title in m.FAMILIAS.items():
        k=fam+'_solo_imagen';j=fam+'_imagen_datos';lo,hi=np.quantile(dist[j]-dist[k],[.025,.975])
        pairs.append(dict(familia=title,BA_solo_imagen=res[k]['exactitud_balanceada'],BA_imagen_datos=res[j]['exactitud_balanceada'],diferencia_BA=res[j]['exactitud_balanceada']-res[k]['exactitud_balanceada'],diferencia_BA_IC95_inferior=lo,diferencia_BA_IC95_superior=hi,AUROC_media_solo_imagen=res[k]['ROC_AUC_media_grupos'],AUROC_media_imagen_datos=res[j]['ROC_AUC_media_grupos']))
    table=pd.DataFrame(rows);pair=pd.DataFrame(pairs)
    table.to_csv(out/'metricas.csv',index=False);pair.to_csv(out/'aporte_variables.csv',index=False)
    shown=table.copy()
    for col in shown:
        if col!='modelo' and not col.startswith('ROC_AUC'):shown[col]*=100
    shown.to_csv(out/'metricas_porcentajes.csv',index=False)
    ps=pair.copy()
    for col in ps:
        if col!='familia' and not col.startswith('AUROC'):ps[col]*=100
    ps.to_csv(out/'aporte_variables_puntos_porcentuales.csv',index=False)
    config=dict(corte_horas=a.corte,max_horas_inclusive=24,edad_minima=18,exclusion_padding=False,cohorte={'imagenes':len(d),'personas':len(people)},semilla=42,logistica_C=.01,svm_C_candidatos=[.1,1,10],bootstrap_BA=nboot,bootstrap_aporte=2000,asignacion='registro suministrado' if a.particiones else 'nueva estratificación por persona',particiones_sha256=sha(a.particiones) if a.particiones else None,versiones={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','Pillow','joblib','matplotlib']})
    (out/'configuracion.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    m.comparar_resultados(out,mostrar=False)
    # Añadir conteos de la referencia, igual que en la ejecución original.
    cm=confusion_matrix(y,bp,labels=[0,1],sample_weight=m.pesos_persona(ids));raw=confusion_matrix(y,bp,labels=[0,1])
    cells=[dict(modelo='Referencia mayoritaria',clase_reportada=m.ETIQUETAS[i],clase_estimada=m.ETIQUETAS[j],numero_lesiones=int(raw[i,j]),peso_personas=cm[i,j],porcentaje_fila_ponderado=100*cm[i,j]/cm[i].sum()) for i in range(2) for j in range(2)]
    f=out/'matrices_conteos_y_porcentajes.csv';pd.concat([pd.DataFrame(cells),pd.read_csv(f)]).to_csv(f,index=False)
    fig,ax=plt.subplots(figsize=(11,7));yy=np.arange(len(table))
    ax.hlines(yy,shown.BA_IC95_inferior,shown.BA_IC95_superior);ax.scatter(shown.exactitud_balanceada,yy)
    ax.set(yticks=yy,yticklabels=table.modelo,xlim=(0,100),xlabel='Exactitud balanceada (%) e IC95%',title=f'Corte {a.corte} h · participantes ≥18 años');ax.invert_yaxis();ax.axvline(50,ls='--',color='gray');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'exactitud_balanceada_todos.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));xx=np.arange(len(pair))
    ax.bar(xx-.18,100*pair.BA_solo_imagen,.36,label='Imagen sola');ax.bar(xx+.18,100*pair.BA_imagen_datos,.36,label='Imagen + datos')
    ax.set(xticks=xx,xticklabels=pair.familia,ylim=(0,100),ylabel='Exactitud balanceada (%)');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'aporte_variables.{ext}',dpi=180)
    plt.close(fig)
    print('Resultados PRIVADOS (incluyen predicciones individuales):',out)
    return out
if __name__=='__main__':main()
