"""Extractores congelados originales: ImageNet y color/textura. No ajustan la red."""
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
SEMILLA=42
LADO=224
N_CARACTERISTICAS=55

def resumir_canal(canal):
    p10, mediana, p90 = np.percentile(canal, [10, 50, 90])
    return [float(canal.mean()), float(canal.std()), float(p10), float(mediana), float(p90)]

def extraer_color_textura(ruta):
    with Image.open(ruta) as original:
        im = ImageOps.exif_transpose(original).convert('RGB')
        # Ajustar el lado mayor a 224 conservando proporción, sin relleno ni recorte extra.
        im = ImageOps.contain(im, (LADO, LADO), method=Image.Resampling.BILINEAR)
        if min(im.size) <= 4:
            raise ValueError(f'Imagen demasiado estrecha para textura LBP: {ruta.name}. Revisar el recorte.')
        rgb = np.asarray(im, dtype=np.float64) / 255.0
    lab = rgb2lab(rgb)
    hsv = rgb2hsv(rgb)
    valores = []
    for canal in [lab[..., 0], lab[..., 1], lab[..., 2], hsv[..., 1], hsv[..., 2]]:
        valores.extend(resumir_canal(canal))
    # Tono circular: rojo cerca de 0 y 1 representa colores próximos.
    # La saturación reduce la influencia de píxeles casi grises; si es cero, tono = 0,0.
    saturacion = hsv[..., 1]
    angulo = 2 * np.pi * hsv[..., 0]
    total_saturacion = float(saturacion.sum())
    if total_saturacion > 1e-12:
        valores.extend([float((saturacion * np.cos(angulo)).sum() / total_saturacion),
                        float((saturacion * np.sin(angulo)).sum() / total_saturacion)])
    else:
        valores.extend([0.0, 0.0])
    gris = img_as_ubyte(rgb2gray(rgb))
    for puntos, radio in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(gris, P=puntos, R=radio, method='uniform')
        # Evitar que el tratamiento de los bordes añada una textura artificial.
        interior = lbp[radio:-radio, radio:-radio].astype(np.int64)
        histograma = np.bincount(interior.ravel(), minlength=puntos + 2).astype(float)
        histograma /= histograma.sum()
        valores.extend(histograma.tolist())
    vector = np.asarray(valores, dtype=np.float64)
    assert vector.shape == (N_CARACTERISTICAS,) and np.isfinite(vector).all()
    return vector

def extraer_imagenes(tipo,rutas):
    if tipo=='color_textura':
        global rgb2lab,rgb2hsv,rgb2gray,local_binary_pattern,img_as_ubyte
        from skimage.color import rgb2lab,rgb2hsv,rgb2gray
        from skimage.feature import local_binary_pattern
        from skimage.util import img_as_ubyte
        return np.stack([extraer_color_textura(p) for p in tqdm(rutas,desc='Color y textura')])
    import torch
    from torchvision.models import resnet18,ResNet18_Weights,efficientnet_b0,EfficientNet_B0_Weights,mnasnet1_0,MNASNet1_0_Weights
    from torchvision import transforms
    torch.manual_seed(SEMILLA);torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMILLA)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    equipo=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if tipo=='resnet':
        red=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);red.fc=torch.nn.Identity();n=512
    elif tipo=='efficientnet':
        red=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1);n=red.classifier[1].in_features;red.classifier=torch.nn.Identity()
    elif tipo=='mnasnet':
        red=mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1);n=red.classifier[1].in_features;red.classifier=torch.nn.Identity()
    else:
        raise ValueError(f'Extractor no reconocido: {tipo}')
    red.requires_grad_(False);red=red.to(equipo).eval()
    normalizar=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def preparar(p):
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            im=ImageOps.pad(im,(LADO,LADO),method=Image.Resampling.BILINEAR,color=(124,116,104))
            return normalizar(im)
    bloques=[]
    with torch.inference_mode():
        for i in tqdm(range(0,len(rutas),16),desc=tipo):
            lote=torch.stack([preparar(p) for p in rutas[i:i+16]])
            bloques.append(red(lote.to(equipo)).cpu().numpy())
    X=np.concatenate(bloques)
    assert X.shape==(len(rutas),n) and np.isfinite(X).all()
    return X

"""Extrae representaciones de recortes de equimosis ya revisados; no realiza selección clínica."""
import argparse
from pathlib import Path
import json
import importlib.metadata
import numpy as np
import pandas as pd
from metodos import base,rutas_imagenes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proyecto',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--modelos',nargs='+',choices=['resnet','efficientnet','mnasnet','color_textura'],default=['resnet','efficientnet','mnasnet','color_textura'])
    a=p.parse_args();d,ex=base(a.proyecto);rutas,hashes=rutas_imagenes(a.proyecto,d)
    a.salida.mkdir(parents=True,exist_ok=False)
    for tipo in a.modelos:
        X=extraer_imagenes(tipo,rutas)
        np.savez_compressed(a.salida/f'{tipo}_caracteristicas.npz',X=X,archivo=d.archivo.to_numpy(str),persona_id=d.persona_id.to_numpy(str))
    d.to_csv(a.salida/'datos_usados.csv',index=False)
    ex.to_csv(a.salida/'excluidos.csv',index=False)
    pd.DataFrame({'archivo':d.archivo,'sha256':hashes}).to_csv(a.salida/'huellas_imagenes.csv',index=False)
    versiones={}
    for k in ['numpy','Pillow','torch','torchvision','scikit-image','tqdm']:
        try:versiones[k]=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:pass
    (a.salida/'configuracion.json').write_text(json.dumps({'versiones':versiones,'redes_congeladas':True,'lado':224,'relleno_RGB':[124,116,104],'pesos':'ImageNet V1','modelos':a.modelos},indent=2))
    print('Características privadas guardadas en',a.salida)
if __name__=='__main__':main()
