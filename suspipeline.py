import pycolmap
import trimesh
import xatlas
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

# FRIST WE RECONSTRUCT WITH COLMAP
output_path: pathlib.Path() # put the paths here
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

# PLACING THE TEXTURE ON THE UVMAPPED MESH WITH MITSUBA 3
mi.set_variant("scalar_rgb")

from mitsuba import ScalarTransform4f as T

scene = mi.load_dict({
    'texture_id':     
    'type': '<texture_type>':
    # .. texture parameters ..
    ,
    'type': 'scene',
    # The keys below correspond to object IDs and can be chosen arbitrarily
    'integrator': {'type': 'path'},
    'light': {'type': 'constant'},
    'cube': {      #put name of enctured matieral 
        'type': 'obj',
        'filename': 'xxx' #insert path to uv maped obj file
        # 'to_world': T.translate([0, 0, -1.5]),
        'bsdf': {
            'type': 'ref',
            'id' : 'texture_id'  #idk if we are suppose to pur the path name here? 
            
       },
    ,
    }
})

#multiangle optimize
mi.set_variant('llvm_ad_rgb')
#do if on nonmac
# mi.set_variant('cuda_ad_rgb')


#scene = mi.load_file('./scenes/cbox.xml', res=128, integrator='prb'
from mitsuba import ScalarTransform4f as T

def load_sensor(r, phi, theta):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 1, 0]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })

sensor_count = 6
radius = 12
phis = [20.0 * i for i in range(sensor_count)]
theta = 10.0

sensors = [load_sensor(radius, phi, theta) for phi in phis]

image_ref = mi.render(scene, spp=512)

# Preview the reference image
mi.util.convert_to_bitmap(image_ref)
plt.xlabel("Default angle reference image")
plt.imshow(image_ref)
plt.show()

#preview multiangle refs
multi_ref = [mi.render(scene, spp=16, sensor=sensor) for sensor in sensors]
count = 0
for image in multi_ref:
    count+=1
    mi.util.convert_to_bitmap(image)
    plt.xlabel("Angle " + str(count) + " reference image")
    plt.imshow(image)
    plt.show()

params = mi.traverse(scene)

key = 'red.reflectance.value'

# Save the original value
param_ref = mi.Color3f(params[key])

# Set another color value and update the scene
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update();

image_init = mi.render(scene, spp=128)
mi.util.convert_to_bitmap(image_init)
plt.xlabel("Default angle initial image")
plt.imshow(image_init)
plt.show()

#preview initial images from diff angles
multi_init = [mi.render(scene, spp=16, sensor=sensor) for sensor in sensors]
count = 0
for image in multi_init:
    count+=1
    mi.util.convert_to_bitmap(image)
    plt.xlabel("Angle " + str(count) + " initial image")
    plt.imshow(image)
    plt.show()

opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt);

#mean does for all things in array, so do sqr for all images and image refs
def mse(image):
    return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50

errors = []

#what they did-took one rendered image, compared it with its original, optimized based on that
# what we want to do-take scene, render from diff camera angles(represents images of thing we take irl), optimize scene params for each image
# in iteration loop, each image angle is compared with its original image angle, parameters are optimized based on that

for it in range(iteration_count):
    # Perform a (noisy) differentiable rendering of the scene 
    # image = mi.render(scene, params, spp=4)

    # # Evaluate the objective function from the current rendered image
    # loss = mse(image)

    # # Backpropagate through the rendering process (derivative of loss function?)
    # dr.backward(loss)

    # # Optimizer: take a gradient descent step - minimize loss function?
    # opt.step()

    # # Post-process the optimized parameters to ensure legal color values.
    # opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    # # Update the scene state to the new optimized values
    # params.update(opt)

    # # Track the difference between the current color and the true value
    err_ref = dr.sum(dr.sqr(param_ref - params[key]))
    print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
    errors.append(err_ref)

    for index in range(sensor_count):
        #DO IT FOR ALL ANGLES
        image = mi.render(scene, params, sensor=sensors[index], spp=4)

        loss = dr.mean(dr.sqr(image - multi_ref[index]))

        dr.backward(loss)

        opt.step()

        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        params.update(opt)
print('\nOptimization complete.')

#do without params? shouldnt matter?
image_final = mi.render(scene, params, spp=128)
mi.util.convert_to_bitmap(image_final)

plt.imshow(image_final)
plt.xlabel("Final image default angle")
plt.show()

# show all images from diff sensor angles
count = 0
for sensor in sensors:
    count += 1
    amogus = mi.render(scene, params, sensor=sensor)
    mi.util.convert_to_bitmap(amogus)
    plt.xlabel("Final image: Angle " + str(count))
    plt.imshow(amogus)
    plt.show()

# plt.plot(errors)
# plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
# plt.show()

# sensor_count = 6

# radius = 12
# phis = [20.0 * i for i in range(sensor_count)]
# theta = 60.0

# sensors = [load_sensor(radius, phi, theta) for phi in phis]

# images = [mi.render(scene, spp=16, sensor=sensor) for sensor in sensors]

# mi.util.convert_to_bitmap(images[1])

# #saves it to desktop
# mi.Bitmap(images[1]).write('image.exr')





