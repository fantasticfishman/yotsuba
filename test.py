
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')

# scene = mi.load_file('./scenes/simple.xml', res=512)
refscene = mi.load_file('./scenes/simplesus.xml')

image_ref = mi.render(refscene, spp=512)

mi.util.convert_to_bitmap(image_ref)
plt.imshow(image_ref)
plt.show()

#config
# render_resolution = (64, 64)
# texture_resolution = (128, 128)
# res = 64
# n_upsampling_steps = spp = 8,
# max_iterations = 25
# learning_rate = .05


# initial_heightmap_resolution = [r // (2**n_upsampling_steps) for r in texture_resolution]
# upsampling_steps = dr.sqr(
#     dr.linspace(mi.Float, 0, 1, n_upsampling_steps + 1, endpoint=False).numpy()[1:])
# upsampling_steps = (max_iterations * upsampling_steps).astype(int)
# print('The resolution of the heightfield will be doubled at iterations:',
#       upsampling_steps)

# construct scene to contain optimized texture

# apply texture to mesh

# blacktext = mi.Bitmap(dr.zeros(mi.TensorXf, [512,512]))
# greytext = mi.Bitmap(dr.full(mi.TensorXf, 0.5, (512,512)))

scene = mi.load_dict({
    "type": "scene",
    "myintegrator": {
        "type": "path",
        "max_depth" : 8
    },
    "meshtext" : {
        "type" : "bitmap",
        "id": "opt_texture",
        "bitmap": mi.Bitmap(dr.full(mi.TensorXf, 0.5, (512,512))),
        "raw": True,
    },
    "testtext": {
        'type': 'bitmap',
        'filename': 'texture1.jpg',
        # 'wrap_mode': 'mirror'
    },
    "mysensor": {
        "type":
        "perspective",
        # "near_clip":
        # 1.0,
        # "far_clip":
        # 1000.0,
        "fov":
        80,
        "to_world":
        mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=60) \
                                    .look_at(target=[0, 0, 0],
                                     origin=[0, 5, 3],
                                     up=[0, 1, 0]),

        "myfilm": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "box"
            },
            "width": 256,
            "height": 256,
        },
        "mysampler": {
            "type": "independent",
            "sample_count": 512,
        },
    },
    "myemitter": {
        "type": "envmap",
        "filename": "./scenes/textures/bank_vault_2k.hdr",
    },
    'testbsdf': {
        'type': 'diffuse',
        'reflectance': {
            'type': 'ref',
            # 'id': 'testtext'
            'id': 'opt_texture'
        }
    },
    "myshape": {
        # "type": "cube",
        "type": "obj",
        "filename" : "./mogusmap.obj",
        # "bsdfred": {
        #     "type": "diffuse",
        #     "reflectance": {
        #         "type": "rgb",
        #         "value": [0.4, 0, 0],
        #         "type" : "bitmap",
        #         'filename' : 'texture1.jpg',
        #     }
        # },
        "bsdftest1": {
            "type": "ref",
            'id' : 'testbsdf'
        },
        "to_world" : mi.ScalarTransform4f.scale(1) \
        .rotate([0,1,0], angle = 45) \
        .rotate([1,0,0], angle = 270)
    }
})


#initialize optomization stuff
sceneparams = mi.traverse(scene)

# this is the scene texture value to be optimized

#declare optimizer 
key = 'opt_texture.data'
dr.enable_grad(sceneparams[key])
#.05 was good
opt = mi.ad.Adam(lr=.05)
opt[key] = sceneparams[key]
sceneparams.update(opt)

#preview initial image
image_opt = mi.render(scene,sceneparams, spp = 512)
mi.util.convert_to_bitmap(image_opt)
plt.imshow(image_opt)
plt.show()

#mse function
def mse(image):
    return dr.mean(dr.sqr(image - image_ref));

iterations = 80
#optomization loop
for it in range(iterations):
    image_opt = mi.render(scene,sceneparams,spp = 100)

    loss = mse(image_opt)

    dr.backward(loss)

    opt.step()

    # opt[key] = dr.clamp(opt[key], 0.0, 1.0)
    # update the data of the texture(does not update scene)
    sceneparams[key] = opt[key]
    sceneparams.update()
    # scene['meshtext'].set_bitmap(mi.Bitmap(sceneparams['opt_texture.data']))
    # update scene texture meshtext.bitmap specifically 
    # scene['meshtext']['bitmap'] = mi.Bitmap(sceneparams[key])
    # if it in (int(0.7 * max_iterations), int(0.9 * max_iterations)):
    #     spp *= 2
    #     opt.set_learning_rate(0.5 * opt.lr[key])

    print("completed iteration " + str(it))
    


image_opt = mi.render(scene, spp=512)

# Preview the final image
mi.util.convert_to_bitmap(image_opt)
plt.imshow(image_opt)
plt.show()
