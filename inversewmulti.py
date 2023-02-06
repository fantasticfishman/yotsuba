import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt




mi.set_variant('llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T

scene = mi.load_file('./scenes/cbox.xml', res=128, integrator='prb')

image_ref = mi.render(scene, spp=512)

# Preview the reference image
mi.util.convert_to_bitmap(image_ref)

#pick the parameter we are optimizing
params = mi.traverse(scene)

key = 'red.reflectance.value'

# Save the original value
param_ref = mi.Color3f(params[key])

# Set another color value and update the scene
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update();

#render the scene again
image_init = mi.render(scene, spp=128)
mi.util.convert_to_bitmap(image_init)
#saves it to desktop
#mi.Bitmap(image_init).write('cbox.exr')


opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
#ensures gradient tracking w respect to wall color
params.update(opt);

#with every iteration of the gradient descent, we will compute the derivates
# of the scene w respect to the obejctive function 

#mean square error btwn the current image and the ref image
def mse(image):
    return dr.mean(dr.sqr(image - image_ref))


#controlling our optimization loop 
iteration_count = 50

#gradient descent loop, itereates the iteration count times

errors = []
for it in range(iteration_count):
    # Perform a (noisy) differentiable rendering of the scene
    image = mi.render(scene, params, spp=4)

    # Evaluate the objective function from the current rendered image
    loss = mse(image)

    # Backpropagate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    # Update the scene state to the new optimized values
    params.update(opt)

    # Track the difference between the current color and the true value
    err_ref = dr.sum(dr.sqr(param_ref - params[key]))
    print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
    errors.append(err_ref)
print('\nOptimization complete.')

def load_sensor(r, phi, theta):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
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

    # we specify the number of samples per signal with spp 

sensor_count = 6

radius = 12
phis = [20.0 * i for i in range(sensor_count)]
theta = 60.0

sensors = [load_sensor(radius, phi, theta) for phi in phis]

images = [mi.render(scene, spp=16, sensor=sensor) for sensor in sensors]

mi.util.convert_to_bitmap(images[1])
#saves it to desktop
mi.Bitmap(images[1]).write('cbox2.exr')


#image_final = mi.render(scene, spp=128)
#mi.util.convert_to_bitmap(image_final)
#mi.Bitmap(image_final).write('cbox.exr')

#plt.plot(errors)
#plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
#plt.show()
