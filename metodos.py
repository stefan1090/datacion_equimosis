"""Funciones verificadas del análisis de 6 horas. Funciones compartidas por los análisis de 6 y 12 horas."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil, unicodedata, warnings, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.exceptions import ConvergenceWarning
import joblib
try:
    from IPython.display import display
except ImportError:
    def display(x): print(x.to_string(index=False) if hasattr(x,'to_string') else x)
SEMILLA=42
C_LOGISTICA=0.01
C_SVM=[0.1,1.0,10.0]
CAT=['localizacion','orientacion','sexo']
META=['localizacion','orientacion','edad','sexo']
ETIQUETAS=['Menos de 6 h','6–24 h']
VERSION_ANALISIS='1.0.0'
CORTE=6
FAMILIAS={'resnet':'ResNet18','efficientnet':'EfficientNet-B0','color_textura':'Color y textura','svm':'EfficientNet + SVM','mnasnet':'MNASNet 1.0'}
warnings.filterwarnings('error',category=ConvergenceWarning)

def pesos_persona(ids):
    conteo=pd.Series(ids).value_counts()
    return np.asarray([1/conteo[p] for p in ids],dtype=float)


def pesos_train(ids,y):
    w=pesos_persona(ids)
    p=pd.DataFrame({'id':ids,'y':y}).drop_duplicates('id')
    n=p.y.value_counts()
    wc=w*np.asarray([len(p)/(2*n[c]) for c in y])
    return w/w.mean(),wc/wc.mean()


def particiones_persona(ids,y,n=5,semilla=SEMILLA):
    tabla=pd.DataFrame({'id':ids,'y':y}).drop_duplicates('id').sort_values('id').reset_index(drop=True)
    if tabla.y.value_counts().reindex([0,1],fill_value=0).min()<n:
        raise ValueError(f'Se necesitan al menos {n} personas por categoría.')
    resultado=[]
    for a,b in StratifiedKFold(n_splits=n,shuffle=True,random_state=semilla).split(tabla.id,tabla.y):
        train=np.flatnonzero(np.isin(ids,tabla.iloc[a].id))
        test=np.flatnonzero(np.isin(ids,tabla.iloc[b].id))
        assert not set(ids[train]) & set(ids[test])
        assert set(y[train])=={0,1} and set(y[test])=={0,1}
        resultado.append((train,test))
    return resultado


def ajustar(X,meta,y,ids,usar_imagen,usar_datos,svm=False,c=C_LOGISTICA):
    """Todos los transformadores se ajustan SOLO con las filas recibidas de entrenamiento."""
    wp,wc=pesos_train(ids,y)
    paquete={'usar_imagen':usar_imagen,'usar_datos':usar_datos,'svm':svm,'columnas_categoricas':CAT.copy()}
    bloques=[]
    if usar_imagen:
        escala=StandardScaler().fit(X,sample_weight=wp)
        bloques.append(escala.transform(X));paquete['escala_imagen']=escala
    if usar_datos:
        escala_edad=StandardScaler().fit(meta[['edad']],sample_weight=wp)
        codificador=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64)
        categorias=codificador.fit_transform(meta[CAT])
        bloques.extend([escala_edad.transform(meta[['edad']]),categorias])
        paquete.update(escala_edad=escala_edad,codificador=codificador)
    Z=np.concatenate(bloques,axis=1)
    clf=SVC(C=c,kernel='rbf',gamma='scale',cache_size=512) if svm else LogisticRegression(C=c,solver='lbfgs',max_iter=5000,random_state=SEMILLA)
    clf.fit(Z,y,sample_weight=wc)
    paquete['clasificador']=clf
    paquete['n_columnas']=Z.shape[1]
    return paquete


def predecir(paquete,X,meta):
    bloques=[]
    if paquete['usar_imagen']:bloques.append(paquete['escala_imagen'].transform(X))
    if paquete['usar_datos']:
        bloques.extend([paquete['escala_edad'].transform(meta[['edad']]),paquete['codificador'].transform(meta[CAT])])
    Z=np.concatenate(bloques,axis=1)
    clf=paquete['clasificador']
    score=clf.decision_function(Z) if paquete['svm'] else clf.predict_proba(Z)[:,1]
    return clf.predict(Z),score


def elegir_c(X,meta,y,ids,usar_datos,semilla):
    divisiones=particiones_persona(ids,y,3,semilla)
    registro=[{'entrenamiento':sorted(set(ids[a])),'validacion':sorted(set(ids[b]))} for a,b in divisiones]
    filas=[]
    for c in C_SVM:
        pred=np.full(len(y),-1,dtype=int)
        for a,b in divisiones:
            paquete=ajustar(X[a],meta.iloc[a],y[a],ids[a],True,usar_datos,True,c)
            pred[b],_=predecir(paquete,X[b],meta.iloc[b])
        assert np.isin(pred,[0,1]).all()
        ba=balanced_accuracy_score(y,pred,sample_weight=pesos_persona(ids))
        filas.append({'C':c,'BA_interna':float(ba)})
    return max(filas,key=lambda fila:fila['BA_interna'])['C'],filas,registro


def evaluar_variante(nombre,familia,X,meta,d,divisiones,usar_imagen,usar_datos,svm,salida):
    y=d.clase.to_numpy();ids=d.persona_id.to_numpy()
    pred=np.full(len(d),-1,dtype=int);score=np.full(len(d),np.nan);folds=np.zeros(len(d),dtype=int)
    ajustes=[]
    for f,(a,b) in enumerate(divisiones,1):
        c=C_LOGISTICA
        if svm:
            c,tabla,internas=elegir_c(X[a],meta.iloc[a],y[a],ids[a],usar_datos,SEMILLA+f)
            assert all(not (set(z['entrenamiento'])|set(z['validacion'])) & set(ids[b]) for z in internas)
            ajustes.append({'grupo_externo':f,'C_elegido':c,'evaluaciones':tabla,'particiones_internas':internas})
        modelo=ajustar(X[a],meta.iloc[a],y[a],ids[a],usar_imagen,usar_datos,svm,c)
        pred[b],score[b]=predecir(modelo,X[b],meta.iloc[b]);folds[b]=f
    assert np.isfinite(score).all() and (folds>0).all() and np.isin(pred,[0,1]).all()
    # Ajuste final independiente de las predicciones utilizadas para evaluar.
    c=C_LOGISTICA
    if svm:
        c,tabla,internas=elegir_c(X,meta,y,ids,usar_datos,SEMILLA+1000)
        ajustes.append({'grupo_externo':'ajuste_final','C_elegido':c,'evaluaciones':tabla,'particiones_internas':internas})
    final=ajustar(X,meta,y,ids,usar_imagen,usar_datos,svm,c)
    final.update(familia=familia,metadata_columnas=META,corte_horas=CORTE,max_horas_inclusive=24,edad_minima=18,etiquetas=ETIQUETAS,preprocesamiento='Ver configuracion.json y código de esta ejecución')
    joblib.dump(final,salida/f'{nombre}_modelo_final.joblib')
    (salida/f'{nombre}_seleccion.json').write_text(json.dumps(ajustes,ensure_ascii=False,indent=2))
    out=d.copy();out['grupo_evaluacion']=folds;out['prediccion']=pred;out['puntuacion']=score
    out['tipo_puntuacion']='margen_svm' if svm else 'probabilidad_no_calibrada'
    out['peso_evaluacion']=pesos_persona(ids)
    out.to_csv(salida/f'{nombre}_predicciones.csv',index=False,encoding='utf-8-sig')
    return pred,score,folds


def metricas(nombre,y,pred,score,ids,folds,boot):
    w=pesos_persona(ids)
    auc_f=[roc_auc_score(y[folds==f],score[folds==f],sample_weight=w[folds==f]) for f in sorted(set(folds))]
    personas=pd.DataFrame({'id':ids,'y':y,'acierto':(y==pred).astype(float)}).groupby('id',sort=True).agg(y=('y','first'),acierto=('acierto','mean'))
    muestras=[]
    for c in [0,1]:
        aciertos=personas.loc[personas.y==c,'acierto'].to_numpy()
        muestras.append(aciertos[boot[c]].mean(axis=1))
    distribucion=(muestras[0]+muestras[1])/2
    lo,hi=np.quantile(distribucion,[.025,.975])
    cm=confusion_matrix(y,pred,labels=[0,1],sample_weight=w)
    return {'modelo':nombre,'exactitud':accuracy_score(y,pred,sample_weight=w),
            'exactitud_balanceada':balanced_accuracy_score(y,pred,sample_weight=w),
            'BA_IC95_inferior':lo,'BA_IC95_superior':hi,
            f'sensibilidad_menos{CORTE}h':cm[0,0]/cm[0].sum(),f'sensibilidad_{CORTE}a24h':cm[1,1]/cm[1].sum(),
            'F1_macro':f1_score(y,pred,average='macro',sample_weight=w,zero_division=0),
            'ROC_AUC':roc_auc_score(y,score,sample_weight=w),'ROC_AUC_media_grupos':np.mean(auc_f)},distribucion


def comparar_resultados(carpeta,mostrar=True):
    """Recalcula ROC y matrices usando una sola ejecución y predicciones fuera de entrenamiento."""
    carpeta=Path(carpeta)
    tabla=pd.read_csv(carpeta/'metricas.csv').set_index('modelo')
    nombres={'solo_datos':'Solo datos (logística)'}
    presentes=[]
    for familia,titulo in FAMILIAS.items():
        claves=[f'{familia}_solo_imagen',f'{familia}_imagen_datos']
        existe=[(carpeta/f'{k}_predicciones.csv').exists() for k in claves]
        if any(existe) and not all(existe):raise ValueError(f'Falta una variante de {familia} en {carpeta}')
        if all(existe):
            presentes.append(familia)
            nombres[claves[0]]=titulo+' · imagen sola'
            nombres[claves[1]]=titulo+' + datos'
    if not presentes:raise ValueError('No hay predicciones de modelos de imagen para comparar.')
    if set(tabla.index)!=(set(nombres.values())|{'Referencia mayoritaria'}):
        raise ValueError('La tabla de métricas no coincide con los modelos de esta carpeta.')
    referencia=pd.read_csv(carpeta/'datos_usados.csv').sort_values('archivo').reset_index(drop=True)
    registro=json.loads((carpeta/'particiones.json').read_text())
    predicciones={};curvas={};por_grupo=[];matrices=[];huellas={}
    resumen=[{'modelo':'Referencia mayoritaria','AUROC_media_grupos':tabla.loc['Referencia mayoritaria','ROC_AUC_media_grupos'],
              'AUROC_agrupada':tabla.loc['Referencia mayoritaria','ROC_AUC'],'numero_grupos':len(registro)}]
    claves_comparacion=['persona_id','archivo','tiempo_horas','clase']+META
    grupos_esperados=None
    for clave,nombre in nombres.items():
        archivo=carpeta/f'{clave}_predicciones.csv'
        p=pd.read_csv(archivo).sort_values('archivo').reset_index(drop=True)
        if p.archivo.duplicated().any() or not p[claves_comparacion].equals(referencia[claves_comparacion]):
            raise ValueError(f'Casos o datos diferentes en {archivo.name}; no mezclar ejecuciones.')
        if not p.tiempo_horas.between(0,24).all() or not ((p.tiempo_horas>=CORTE).astype(int)==p.clase).all():
            raise ValueError(f'Criterio temporal incompatible en {archivo.name}')
        if p.groupby('persona_id').grupo_evaluacion.nunique().max()!=1:
            raise ValueError('Una persona tiene imágenes en más de una partición.')
        if grupos_esperados is None:grupos_esperados=p.grupo_evaluacion.copy()
        elif not p.grupo_evaluacion.equals(grupos_esperados):raise ValueError('Particiones diferentes entre modelos.')
        for particion in registro:
            entrenamiento=set(particion['personas_entrenamiento']);evaluacion=set(particion['personas_evaluacion'])
            if entrenamiento&evaluacion or entrenamiento|evaluacion!=set(p.persona_id):raise ValueError('Partición por persona inválida.')
            if set(p.loc[p.grupo_evaluacion==particion['grupo'],'persona_id'])!=evaluacion:raise ValueError('Predicciones no coinciden con las particiones guardadas.')
        w=pesos_persona(p.persona_id.to_numpy())
        if not np.allclose(w,p.peso_evaluacion) or not np.isfinite(p.puntuacion).all():raise ValueError('Pesos o puntuaciones inválidos.')
        auc=roc_auc_score(p.clase,p.puntuacion,sample_weight=w)
        ba=balanced_accuracy_score(p.clase,p.prediccion,sample_weight=w)
        curva=[];valores=[]
        for grupo,g in p.groupby('grupo_evaluacion'):
            fpr,tpr,_=roc_curve(g.clase,g.puntuacion,sample_weight=g.peso_evaluacion)
            a=roc_auc_score(g.clase,g.puntuacion,sample_weight=g.peso_evaluacion)
            curva.append((fpr,tpr));valores.append(a)
            por_grupo.append({'modelo':nombre,'grupo_evaluacion':int(grupo),'AUROC':a,'personas':g.persona_id.nunique(),'imagenes':len(g)})
        for col,valor in [('exactitud_balanceada',ba),('ROC_AUC',auc),('ROC_AUC_media_grupos',np.mean(valores))]:
            if not np.isclose(valor,tabla.loc[nombre,col]):raise ValueError(f'{col} no coincide con las predicciones de {nombre}.')
        resumen.append({'modelo':nombre,'AUROC_media_grupos':np.mean(valores),'AUROC_agrupada':auc,'numero_grupos':len(valores)})
        cm=confusion_matrix(p.clase,p.prediccion,labels=[0,1],sample_weight=w)
        raw=confusion_matrix(p.clase,p.prediccion,labels=[0,1])
        for i in range(2):
            for j in range(2):matrices.append({'modelo':nombre,'clase_reportada':ETIQUETAS[i],'clase_estimada':ETIQUETAS[j],'numero_lesiones':int(raw[i,j]),'peso_personas':cm[i,j],'porcentaje_fila_ponderado':100*cm[i,j]/cm[i].sum()})
        predicciones[clave]=p;curvas[clave]=curva;huellas[archivo.name]=hashlib.sha256(archivo.read_bytes()).hexdigest()
    def guardar(fig,nombre):
        fig.savefig(carpeta/f'{nombre}.png',dpi=180,bbox_inches='tight')
        fig.savefig(carpeta/f'{nombre}.pdf',bbox_inches='tight')
        if mostrar:plt.show()
        plt.close(fig)
    colores={'solo_datos':'#111827','resnet':'#0072B2','efficientnet':'#009E73','color_textura':'#D55E00','svm':'#CC79A7','mnasnet':'#8B4513'}
    grid=np.linspace(0,1,1001)
    for agrupada in [False,True]:
        fig,ejes=plt.subplots(1,2,figsize=(14,6.5))
        for ax,sufijo,titulo in zip(ejes,['solo_imagen','imagen_datos'],['Imagen sola y control con datos','Imagen + datos y control con datos']):
            for clave in ['solo_datos']+[f'{f}_{sufijo}' for f in presentes]:
                familia=clave.removesuffix('_'+sufijo)
                nombre=nombres[clave];p=predicciones[clave]
                if agrupada:
                    x,y,_=roc_curve(p.clase,p.puntuacion,sample_weight=p.peso_evaluacion)
                    a=tabla.loc[nombre,'ROC_AUC']
                else:
                    x=grid;y=np.mean([np.interp(grid,xx,yy) for xx,yy in curvas[clave]],axis=0)
                    a=tabla.loc[nombre,'ROC_AUC_media_grupos']
                ax.plot(x,y,color=colores[familia],lw=2.5 if clave=='solo_datos' else 1.7,label=f'{nombre}: {a:.3f}')
            ax.plot([0,1],[0,1],':',color='gray',label='Sin discriminación: 0.500')
            ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='1 − especificidad',ylabel=f'Sensibilidad ({CORTE}–24 h)',title=titulo)
            ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.15)
        titulo='ROC con puntuaciones agrupadas' if agrupada else 'ROC media entre particiones por persona'
        fig.suptitle(titulo)
        fig.text(.5,.015,'Mismos casos y particiones. Igual peso por persona. '+('Las escalas entre particiones pueden diferir, especialmente en SVM.' if agrupada else 'Leyendas: media de las AUROC exactas por partición; curvas interpoladas.'),ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.94));guardar(fig,'ROC_agrupada_complementaria' if agrupada else 'ROC_todos_los_modelos')
    entradas=[('Referencia mayoritaria',np.zeros(len(referencia),dtype=int))]+[(nombres[k],p.prediccion.to_numpy()) for k,p in predicciones.items()]
    cols=3;filas=int(np.ceil(len(entradas)/cols))
    fig,ejes=plt.subplots(filas,cols,figsize=(12,3.8*filas),squeeze=False)
    for ax,(titulo,pr) in zip(ejes.flat,entradas):
        if titulo=='Referencia mayoritaria':
            for part in registro:
                train=referencia[referencia.persona_id.isin(part['personas_entrenamiento'])].drop_duplicates('persona_id')
                pr[referencia.persona_id.isin(part['personas_evaluacion'])]=int(train.clase.mean()>.5)
        cm=confusion_matrix(referencia.clase,pr,labels=[0,1],sample_weight=pesos_persona(referencia.persona_id.to_numpy()))
        cm=cm/cm.sum(axis=1,keepdims=True)
        ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{100*cm[i,j]:.1f} %',ha='center',va='center',color='white' if cm[i,j]>.55 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=ETIQUETAS,yticklabels=ETIQUETAS,xlabel='Estimado',ylabel='Reportado',title=titulo)
        ax.title.set_fontsize(10)
    for ax in list(ejes.flat)[len(entradas):]:ax.set_visible(False)
    fig.suptitle('Matrices de confusión de todos los modelos',y=1)
    fig.tight_layout();guardar(fig,'matrices_todos_los_modelos')
    pd.DataFrame(resumen).to_csv(carpeta/'resumen_AUROC.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(por_grupo).to_csv(carpeta/'AUROC_por_particion.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(matrices).to_csv(carpeta/'matrices_conteos_y_porcentajes.csv',index=False,encoding='utf-8-sig')
    (carpeta/'fuentes_comparacion.json').write_text(json.dumps({'version':VERSION_ANALISIS,'predicciones_sha256':huellas,'personas':referencia.persona_id.nunique(),'imagenes':len(referencia)},indent=2))
    (carpeta/'LEEME_COMPARACION.txt').write_text('Comparación de una sola ejecución; incluye todos sus modelos aprendidos y la referencia mayoritaria en las matrices.\nAUROC: se muestran media entre particiones y agrupada, que no son intercambiables. No son porcentajes de aciertos. La referencia sin discriminación de las ROC es 0.50.\nNo se mezclan ejecuciones anteriores ni resultados con otra cohorte. No se calculan IC de AUROC aquí. Los IC de exactitud balanceada en metricas.csv son aproximados sobre predicciones fijas.\nMNASNet 1.0 utiliza pesos de ImageNet y un extractor congelado: no replica el sistema AutoML del artículo de Tirado y Mauricio.\n',encoding='utf-8')
    print('Comparación actualizada:',carpeta)
    return pd.DataFrame(resumen)


def check(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def definir_cohorte(original):
    d=original.copy()
    required=['persona_id','archivo','tiempo_horas']+META
    check(set(required).issubset(d.columns),'Faltan columnas necesarias en datos_usados.csv.')
    check(not d[required].isna().any().any(),'Faltan datos; no se imputan valores para este análisis.')
    for col in ['tiempo_horas','edad']:
        d[col]=pd.to_numeric(d[col],errors='raise')
        check(np.isfinite(d[col]).all(),f'{col}: datos no finitos.')
    check(d.tiempo_horas.ge(0).all() and d.edad.between(0,120).all(),'Edades o tiempos inválidos.')
    check(not d.archivo.duplicated().any(),'Archivos duplicados.')
    for col in ['persona_id','archivo']+CAT:
        d[col]=d[col].map(lambda v:unicodedata.normalize('NFC',str(v).strip()))
        check(d[col].ne('').all(),f'{col}: valores vacíos.')
    check(not d.archivo.duplicated().any(),'Archivos duplicados tras normalizar nombres.')
    for col in ['edad','sexo','tiempo_horas']:
        check(d.groupby('persona_id')[col].nunique().max()==1,f'{col} inconsistente dentro de una persona.')
    reason=np.where(d.edad<18,'edad menor de 18 años','')
    reason=np.where(d.tiempo_horas>24,np.where(reason!='',reason+'; tiempo mayor de 24 h','tiempo mayor de 24 h'),reason)
    excluded=d.loc[reason!='',required].copy();excluded['motivo']=reason[reason!='']
    d=d.loc[reason=='',required].sort_values(['persona_id','archivo']).reset_index(drop=True)
    check(len(d)>0,'No hay participantes elegibles.')
    d['clase']=(d.tiempo_horas>=CORTE).astype(int)
    return d,excluded




def configurar(corte):
    """Configura el desenlace sin cambiar el procedimiento estadístico."""
    global CORTE, ETIQUETAS
    if corte not in (6,12):raise ValueError('El corte debe ser 6 o 12 horas.')
    CORTE=corte
    ETIQUETAS=[f'Menos de {corte} h',f'{corte}–24 h']

"""Validación de datos privados, características y particiones por persona."""
from pathlib import Path
import hashlib
import json
import unicodedata
import numpy as np
import pandas as pd
from PIL import Image

DIMENSIONES={'resnet':512,'efficientnet':1280,'color_textura':55,'mnasnet':1280}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def base(proyecto,corte=6):
    original=pd.read_csv(Path(proyecto)/'base_analitica.csv',dtype={'persona_id':str,'archivo':str})
    d,excluidos=definir_cohorte(original)
    d['clase']=(d.tiempo_horas>=corte).astype(int)
    # Solo nombres de archivo: impide rutas fuera de la carpeta de imágenes.
    if any(Path(x).name!=x or x in {'.','..'} or '\\' in x for x in d.archivo):
        raise ValueError('archivo debe ser un nombre simple, sin rutas.')
    return d,excluidos

def rutas_imagenes(proyecto,d):
    index={}
    for p in (Path(proyecto)/'imagenes').iterdir():
        if p.is_file():index.setdefault(unicodedata.normalize('NFC',p.name),[]).append(p)
    rutas=[]
    for nombre in d.archivo:
        if len(index.get(nombre,[]))!=1:raise ValueError('Falta una imagen o hay un nombre ambiguo: '+nombre)
        ruta=index[nombre][0]
        with Image.open(ruta) as im:im.verify()
        rutas.append(ruta)
    hashes=[sha(p) for p in rutas]
    if len(hashes)!=len(set(hashes)):raise ValueError('Imágenes idénticas duplicadas.')
    return rutas,hashes

def cargar(proyecto,caracteristicas,d,familias):
    p=Path(caracteristicas)
    rutas,hashes=rutas_imagenes(proyecto,d)
    huellas=pd.read_csv(p/'huellas_imagenes.csv',dtype=str)
    if huellas.archivo.duplicated().any():raise ValueError('Huellas duplicadas.')
    referencia=huellas.set_index('archivo').sha256
    if not set(d.archivo)<=set(referencia.index) or hashes!=referencia.loc[d.archivo].tolist():
        raise ValueError('Las imágenes cambiaron desde la extracción de características.')
    cache={}
    for key in sorted(set('efficientnet' if k=='svm' else k for k in familias)):
        with np.load(p/f'{key}_caracteristicas.npz',allow_pickle=False) as z:
            files=z['archivo'].astype(str);ids=z['persona_id'].astype(str);X=z['X']
        if len(set(files))!=len(files):raise ValueError('Características duplicadas.')
        order=pd.Index(files).get_indexer(d.archivo)
        if (order<0).any() or not np.array_equal(ids[order],d.persona_id.to_numpy()):
            raise ValueError('Características sin correspondencia con imágenes/personas.')
        if X.shape!=(len(files),DIMENSIONES[key]) or not np.isfinite(X).all():
            raise ValueError('Dimensión o valores inválidos en '+key)
        cache[key]=X[order]
    return cache,pd.DataFrame({'archivo':d.archivo,'sha256':hashes})

def particiones(path,d):
    """Acepta el registro original y retira solo personas no elegibles de esa cohorte."""
    registro=json.loads(Path(path).read_text())
    ids=d.persona_id.to_numpy();y=d.clase.to_numpy();kept=set(ids);seen=set();result=[];universo=None
    if len(registro)!=5 or [r['grupo'] for r in registro]!=[1,2,3,4,5]:
        raise ValueError('Se necesitan cinco particiones numeradas en orden de 1 a 5.')
    for r in registro:
        a=set(r['personas_entrenamiento']);b=set(r['personas_evaluacion'])
        if universo is None:universo=a|b
        if a&b or a|b!=universo or b&seen:raise ValueError('Particiones inconsistentes o solapadas.')
        seen|=b
        tr=np.flatnonzero(np.isin(ids,list(a)));te=np.flatnonzero(np.isin(ids,list(b)))
        if set(y[tr])!={0,1} or set(y[te])!={0,1}:raise ValueError('Partición sin ambas clases.')
        result.append((tr,te))
    if seen!=universo or not kept<=universo:raise ValueError('Cobertura incompleta de participantes.')
    return result
