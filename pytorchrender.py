

import os
import sys
import torch

import matplotlib.pyplot as plt

from pytorch3d.io import load_objs_as_meshes, load_obj

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

sys.path.append(os.path.abspath(''))

from plot_image_grid import image_grid

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


obj_filename = "./cow.obj"
# obj_filename = "./glassmogus.obj"


mesh = load_objs_as_meshes([obj_filename], device=device)

plt.figure(figsize=(7,7))
texturesuv_image_matplotlib(mesh.textures, subsample=None)
plt.axis("off");

R, T = look_at_view_transform(2.7, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

rendered_image = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(rendered_image[0, ..., :3].cpu().numpy())
plt.axis("off");

lights.location = torch.tensor([0.0, 0.0, +1.0], device=device)[None]
images = renderer(mesh, lights=lights)

plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");

R, T = look_at_view_transform(dist=2.7, elev=10, azim=-150)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)

materials = Materials(
    device=device,
    specular_color=[[0.0, 1.0, 0.0]],
    shininess=10.0
)

images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)

plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");

plt.show()

R, T = look_at_view_transform(dist=2.7, elev=10, azim=-150)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
 
lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)
 
# Set material properties
ambient_color = torch.tensor([[1, 1, 1]])  # Ambient reflectivity (low value)
diffuse_color = torch.tensor([[1, 1, 1]])  # Diffuse reflectivity
specular_color = torch.tensor([[3.0, 3.0, 3.0]])  # Specular reflectivity (high value)
shininess = torch.tensor([1000.0])  # Specular exponent (high value)

# Create a Materials object
materials = Materials(
    ambient_color=ambient_color,
    diffuse_color=diffuse_color,
    specular_color=specular_color,
    shininess=shininess,
)
 
images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");

plt.show()