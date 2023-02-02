import pycolmap
reconstruction = pycolmap.Reconstruction("./gerrard-hall/sparse")
print(reconstruction.summary())

for image_id, image in reconstruction.images.items():
    print(image_id, image)

for point3D_id, point3D in reconstruction.points3D.items():
    print(point3D_id, point3D)

for camera_id, camera in reconstruction.cameras.items():
    print(camera_id, camera)

reconstruction.write("./")
reconstruction.export_PLY("sus.ply")
