"""Convierte la hoja Dataset del Excel curado en el CSV de entrada, sin publicar datos."""
import argparse
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

COLUMNAS=['persona_id','archivo','tiempo_horas','edad','sexo','localizacion','cara']
FECHAS=['fecha_hora_lesion','fecha_hora_captura']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--excel',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True,help='Carpeta privada nueva.')
    a=p.parse_args()
    d=pd.read_excel(a.excel,sheet_name='Dataset')
    requeridas=COLUMNAS+FECHAS
    if not set(requeridas)<=set(d.columns):raise ValueError('Faltan columnas en la hoja Dataset.')
    for c in ['persona_id','archivo','localizacion','cara','sexo']:
        if d[c].isna().any():raise ValueError(f'Faltan datos en {c}.')
        d[c]=d[c].map(lambda x:unicodedata.normalize('NFC',str(x).strip()))
        if d[c].eq('').any():raise ValueError(f'Hay textos vacíos en {c}.')
    d['localizacion']=d.localizacion.str.lower()
    d['cara']=d.cara.str.lower()
    d['sexo']=d.sexo.str.upper()
    d['edad']=pd.to_numeric(d.edad,errors='raise')
    d['tiempo_horas']=pd.to_numeric(d.tiempo_horas,errors='raise')
    if d.archivo.duplicated().any() or not d.edad.between(0,120).all():
        raise ValueError('Archivos duplicados o edad inválida.')
    if 'lesion_id' in d and d.lesion_id.duplicated().any():raise ValueError('lesion_id duplicado.')
    if not d.sexo.isin(['M','F']).all():raise ValueError('Sexo fuera de las categorías registradas.')
    for c in ['edad','sexo','tiempo_horas']:
        if d.groupby('persona_id')[c].nunique(dropna=False).max()>1:
            raise ValueError(f'{c} difiere entre imágenes de una persona.')
    lesion=pd.to_datetime(d.fecha_hora_lesion,errors='raise')
    foto=pd.to_datetime(d.fecha_hora_captura,errors='raise')
    calculado=(foto-lesion).dt.total_seconds()/3600
    if not np.array_equal(calculado.notna(),d.tiempo_horas.notna()):
        raise ValueError('Los tiempos faltantes difieren de las fechas.')
    if not np.allclose(calculado.dropna(),d.tiempo_horas.dropna(),atol=1e-7,rtol=0):
        raise ValueError('El tiempo reportado difiere del calculado con las fechas.')
    if calculado.dropna().lt(0).any():raise ValueError('Tiempos negativos.')
    d['tiempo_horas']=calculado
    d=d.rename(columns={'cara':'orientacion'})
    cols=['persona_id','archivo','tiempo_horas','edad','sexo','localizacion','orientacion']
    # La selección clínica y los recortes de calidad ya deben estar resueltos en este Excel.
    # Se retiran solo filas sin tiempo; edad ≥18 y tiempo ≤24 se aplican en analizar.py.
    elegibles=d.loc[d.tiempo_horas.notna(),cols].sort_values(['persona_id','archivo'])
    a.salida.mkdir(parents=True,exist_ok=False)
    elegibles.to_csv(a.salida/'base_analitica.csv',index=False,encoding='utf-8-sig')
    print(f'Base privada guardada: {len(elegibles)} imágenes de {elegibles.persona_id.nunique()} personas; {int(d.tiempo_horas.isna().sum())} filas sin tiempo omitidas.')
if __name__=='__main__':main()
