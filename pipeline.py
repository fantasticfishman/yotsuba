#combining colmap and xatlas
import pycolmap
import trimesh
import xatlas
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

# FRIST WE RECONSTRUCT WITH COLMAP
output_path: pathlib.Path()  # put the paths here
image_dir: pathlib.Path()

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# dense reconstruction
pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

# THEN WE GET XATLAS TO DO UV MAPPING
mesh = trimesh.load_mesh("meshed-poisson.ply")
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
xatlas.export("output.obj", mesh.vertices[vmapping], indices, uvs)