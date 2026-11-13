"""Build IACP canonical assets on CPU from public InteractVLM definitions.
The result is explicitly not claimed byte-identical to unpublished binaries.
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
 return h.hexdigest()

def build(output,body_model_path):
 import numpy as np
 import torch
 from PIL import Image
 from gcanet.compat import prepare_legacy_smpl
 prepare_legacy_smpl()
 from smplx import build_layer
 from pytorch3d.structures import Meshes
 from pytorch3d.renderer import (TexturesVertex,look_at_view_transform,FoVPerspectiveCameras,RasterizationSettings,MeshRasterizer,MeshRenderer,HardPhongShader,PointLights)
 torch.set_num_threads(4)
 output=Path(output); output.mkdir(parents=True,exist_ok=False)
 body=build_layer(model_path=str(body_model_path),model_type='smplh',use_pca=False,gender='neutral',ext='pkl',num_betas=10)
 rotations=torch.eye(3).repeat(1,21,1,1)
 for index,degrees in [(0,30.),(1,-30.)]:
  a=torch.tensor(degrees*np.pi/180); c,s=a.cos(),a.sin()
  rotations[0,index]=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]])
 with torch.no_grad(): vertices=body(body_pose=rotations).vertices[0]
 faces=torch.from_numpy(body.faces.astype(np.int64))
 if vertices.shape!=(6890,3) or faces.min()!=0 or faces.max()!=6889: raise ValueError('Expected 6890-vertex SMPL-H topology')
 normals=torch.zeros_like(vertices)
 fn=torch.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]],dim=1)
 for corner in range(3): normals.index_add_(0,faces[:,corner],fn)
 normals=torch.nn.functional.normalize(normals,p=2,dim=1)
 mesh=Meshes(verts=[vertices],faces=[faces],textures=TexturesVertex(verts_features=((normals+1)/2)[None]))
 # Original dict order and index-dependent light locations are preserved.
 generation=[('topfront',[2,45,315,0,0],[0,0,3]),('topback',[2,45,135,0,0],[0,0,-3]),('bottomfront',[2,315,315,0,.3],[0,0,-3]),('bottomback',[2,315,135,0,.3],[0,0,3])]
 indices,barycentrics,details={},{},[]
 for name,params,light in generation:
  d,elev,azim,tx,ty=params
  R,T=look_at_view_transform(d,elev,azim,device='cpu'); T[0,0]+=tx; T[0,1]+=ty
  camera=FoVPerspectiveCameras(device='cpu',R=R,T=T)
  rasterizer=MeshRasterizer(cameras=camera,raster_settings=RasterizationSettings(image_size=(1024,1024),blur_radius=0.,faces_per_pixel=1))
  with torch.no_grad(): fragments=rasterizer(mesh)
  fidx=fragments.pix_to_face[0,...,0]; valid=fidx>=0
  p2v=torch.full((1024,1024,3),-1,dtype=torch.int64); p2v[valid]=faces[fidx[valid]]
  bary=fragments.bary_coords[0,...,0,:]; err=(bary[valid].sum(-1)-1).abs().max().item()
  if err>2e-5 or bary[valid].min() < -2e-5: raise ValueError('Invalid barycentrics: '+name)
  lights=PointLights(device='cpu',ambient_color=[[.5,.5,.5]],diffuse_color=[[.3,.3,.3]],specular_color=[[.2,.2,.2]],location=[light])
  renderer=MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(device='cpu',cameras=camera,lights=lights))
  with torch.no_grad(): rgb=renderer(mesh)[0,...,:3]
  Image.fromarray((rgb.numpy()*255).astype(np.uint8)).save(output/f'body_render_norm_{name}.png')
  indices[name]=p2v.numpy(); barycentrics[name]=bary.numpy()
  details.append(dict(name=name,camera=params,light=light,camera_R=R.tolist(),camera_T=T.tolist(),valid_pixels=int(valid.sum()),supported_vertices=int(torch.unique(p2v[valid]).numel()),barycentric_sum_max_error=err))
  print(json.dumps(details[-1]),flush=True)
 np.savez_compressed(output/'pixel_to_vertex_map_1024.npz',**indices)
 np.savez_compressed(output/'bary_coords_map_1024.npz',**barycentrics)
 np.savez_compressed(output/'canonical_mesh.npz',vertices=vertices.numpy(),faces=faces.numpy(),body_pose=rotations.numpy())
 files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()]
 manifest=dict(schema='pihoc_canonical_reconstruction_v1',origin='reconstructed_from_published_InteractVLM_geometry_and_rendering_definitions',byte_exact_to_unreleased_original_assets=False,device='cpu',body_model=dict(path=str(Path(body_model_path).resolve()),sha256=sha(body_model_path)),source_urls=['https://github.com/saidwivedi/InteractVLM/blob/main/preprocess_data/render_mesh_utils.py','https://github.com/saidwivedi/InteractVLM/blob/main/preprocess_data/generate_damon_human_mask.py'],generator_sha256=sha(__file__),inference_view_order=['topfront','bottomfront','topback','bottomback'],generation_views=details,vertices=6890,faces=len(faces),topology_sha256=hashlib.sha256(faces.numpy().tobytes()).hexdigest(),files=files,limitations=['Original binary assets and exact original renderer version/model bytes unavailable; no byte-exact original Pi-HOC output claim'])
 (output/'canonical_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(output/'canonical_manifest.json')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--body-model',required=True)
 a=p.parse_args();build(a.output,a.body_model)
