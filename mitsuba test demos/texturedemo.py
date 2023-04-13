#mitsuba demo of texture optomization
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

#get model that is not uv mapped

import trimesh
import xatlas

#multiangle optimize
mi.set_variant('llvm_ad_rgb')
#do if on nonmac
# mi.set_variant('cuda_ad_rgb')

#get test mesh and map it

mesh1 = trimesh.load_mesh("amogus.ply")

atlas = xatlas.Atlas()

vmapping, indices, uvs = xatlas.parametrize(mesh1.vertices, mesh1.faces)

# Optionally parametrize the generation with
# `xatlas.ChartOptions` and `xatlas.PackOptions`.
xatlas.export("output.obj", mesh1.vertices[vmapping], indices, uvs)

#config
render_resolution = (64, 64)
texture_resolution = (128, 128)
res = 128
n_upsampling_steps = 4
spp = 8,
max_iterations = 25
learning_rate = 3e-5


initial_heightmap_resolution = [r // (2**n_upsampling_steps) for r in texture_resolution]
upsampling_steps = dr.sqr(
    dr.linspace(mi.Float, 0, 1, n_upsampling_steps + 1, endpoint=False).numpy()[1:])
upsampling_steps = (max_iterations * upsampling_steps).astype(int)
print('The resolution of the heightfield will be doubled at iterations:',
      upsampling_steps)

#create blank texture bitmap
meshtext = mi.load_dict({
    'type':
    'bitmap',
    'id':
    'mesh_texture',
    'bitmap':
    mi.Bitmap(dr.zeros(mi.TensorXf, initial_heightmap_resolution)),
    'raw':
    True,
})

# Actually optimized: the heightmap texture
meshparams = mi.traverse(meshtext)
meshparams.keep(['data'])
opt = mi.ad.Adam(lr=3e-5, params=meshparams)

#apply texture to output mesh

#this is done in the opt amongbox xml

scene = mi.load_dict({
    "type": "scene",
    "integrator": {
        "type": "path",
    },
    "mysensor": {
        "type":
        "perspective",
        "near_clip":
        1.0,
        "far_clip":
        1000.0,
        "to_world":
        mi.ScalarTransform4f.look_at(origin=[1, 1, 1],
                                     target=[0, 0, 0],
                                     up=[0, 0, 1]),
        "mysampler": {
            "type": "independent",
            "sample_count": 4,
        },
    },
    "myemitter": {
        "type": "constant"
    },
    "mytexture": {
        "type": "ref",
        "id": meshtext,
    },
    "myobject": {
        "type": "obj",
        "filename": "output.obj",
        "bsdf": "mytexture",
    }
})

refscene = mi.load_file('./scenes/ref_among_box.xml')
refimg = mi.render(refscene)
scene = mi.load_file('./scenes/opt_among_box.xml')

sceneparams = mi.traverse(scene)

iterations = 1000
spp = 8
for it in range(iterations):

    #apply displacement?
    dr.enable_grad(meshparams['data'])
    sceneparams.update()

    #render
    image = mi.render(scene, meshparams, seed=it, spp=2 * spp, spp_grad=spp)
    #loss

    loss = dr.mean(dr.sqr(image - refimg))

    #backstep(loss)
    dr.backward(loss)

    #step (no params?)
    opt.step()

    #increase resolution

    if it in upsampling_steps:
        opt['data'] = dr.upsample(opt['data'], scale_factor=(2, 2, 1))

    meshparams.update(opt)

    # Increase rendering quality toward the end of the optimization
    if it in (int(0.7 * iterations), int(0.9 * iterations)):
        spp *= 2
        opt.set_learning_rate(0.5 * opt.lr['data'])
sceneparams.update()

#render
image = mi.render(scene, meshparams, seed=it, spp=2 * spp, spp_grad=spp)

mi.util.convert_to_bitmap(image)
plt.imshow(image)
plt.show()