import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

scene = mi.load_file('./scenes/cbox.xml', res=128, integrator='prb')

image_ref = mi.render(scene, spp=512)

# Preview the reference image
mi.util.convert_to_bitmap(image_ref)

params = mi.traverse(scene)

key = 'red.reflectance.value'

# Save the original value
param_ref = mi.Color3f(params[key])

# Set another color value and update the scene
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update();

image_init = mi.render(scene, spp=128)
mi.util.convert_to_bitmap(image_init)

opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt);

#mean does for all things in array, so do sqr for all images and image refs
def mse(image):
    return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50

errors = []

# for what we want to do, get sensors, then do images = render for sensor in sensors for every time you do render
# then at the end all images should be fixed 
for it in range(iteration_count):
    # Perform a (noisy) differentiable rendering of the scene - do it as images instead
    image = mi.render(scene, params, spp=4)

    # Evaluate the objective function from the current rendered image -do it for all the images? make sure you end up with one number
    loss = mse(image)

    # Backpropagate through the rendering process (derivative of loss function?)
    dr.backward(loss)

    # Optimizer: take a gradient descent step - minimize loss function?
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

image_final = mi.render(scene, spp=128)
mi.util.convert_to_bitmap(image_final)

plt.plot(errors)
plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
plt.show()