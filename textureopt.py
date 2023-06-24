import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')

# This script demos optimizing a blank texture to anything else using gradient descent.
#the first image displayed is the mesh with a texture applied(representing ground truth)
# second image is blank texture
# after optomization final image with optomized texture is displayed

# scene = mi.load_file('./scenes/simple.xml', res=512)
#identical scene with red texture
# refscene = mi.load_file('./scenes/simplesus.xml')

# Create reference scene with truth texture
refscene = mi.load_dict({
    "type": "scene",
    "myintegrator": {
        "type": "path",
        "max_depth" : 8
    },
    "testtext": {
        'type': 'bitmap',
        "id": "testtext",
        # 'filename': 'texture1.jpg',
        'filename': './scenes/textures/texture2.png',
        # 'wrap_mode': 'mirror'
    },
    "mysensor": {
        "type":
        "perspective",
        "fov":
        80,
        "to_world":
        mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=60) \
                                    .look_at(target=[0, 0, 0],
                                     origin=[0, 17, 3],
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
            'id': 'testtext'
            # 'id': 'opt_texture'
        }
    },
    "myshape": {
        "type": "obj",
        "filename" : "./scenes/meshes/mogusmap.obj",
        "bsdftest1": {
            "type": "ref",
            'id' : 'testbsdf'
        },
        "to_world" : mi.ScalarTransform4f.scale(5) \
        .rotate([0,1,0], angle = 45) \
        .rotate([1,0,0], angle = 270)
    }
})

image_ref = mi.render(refscene, spp=512)

#preview reference image
mi.util.convert_to_bitmap(image_ref)
plt.imshow(image_ref)
plt.show()

# construct identical scene with blank texture

# apply texture to mesh

scene = mi.load_dict({
    "type": "scene",
    "myintegrator": {
        "type": "path",
        "max_depth" : 8
    },
    "meshtext" : {
        "type" : "bitmap",
        "id": "opt_texture",
        "bitmap": mi.Bitmap(dr.full(mi.TensorXf, 0.5, (256,256,3))),
        "raw": True,
    },
    "mysensor": {
        "type":
        "perspective",
        "fov":
        80,
        "to_world":
        mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=60) \
                                    .look_at(target=[0, 0, 0],
                                     origin=[0, 17, 3],
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
    # "myemitter": {
    #     'type': 'constant',
    #     'radiance': {
    #         'type': 'rgb',
    #         'value': 1.0,
    #     }
    # },
    'testbsdf': {
        'type': 'diffuse',
        'reflectance': {
            'type': 'ref',
            # 'id': 'testtext'
            'id': 'opt_texture'
        }
    },
    "myshape": {
        "type": "obj",
        "filename" : "./scenes/meshes/mogusmap.obj",
        "bsdftest1": {
            "type": "ref",
            'id' : 'testbsdf'
        },
        "to_world" : mi.ScalarTransform4f.scale(5) \
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
opt = mi.ad.Adam(lr=1e-2)
opt[key] = sceneparams[key]
sceneparams.update(opt)

#preview initial image
image_opt = mi.render(scene, sceneparams, spp=512)
# mi.util.convert_to_bitmap(image_opt)
plt.imshow(image_opt)
plt.show()


#mse function
def mse(image):
    return dr.mean(dr.sqr(image - image_ref))

#optomization loop
iterations = 80
spp = 16

#start high and decrease learning rate? nah
for it in range(iterations):
    #increase quality and decrease learning rate over time
    # if(it % 10 == 0):
    #     spp = spp * 2
    #     opt.set_learning_rate(0.5 * opt.lr[key])
    image_opt = mi.render(scene, sceneparams, spp=spp)
    # plt.imshow(image_ref)
    # plt.show()
    # plt.imshow(image_opt)
    # plt.show()

    loss = mse(image_opt)
    print(loss)

    dr.backward(loss)

    opt.step()

    # opt[key] = dr.clamp(opt[key], 0.0, 1.0)
    # update the data of the texture(does not update scene)
    sceneparams[key] = dr.clamp(opt[key], 0.0, 1.0)
    sceneparams.update()

    print("completed iteration " + str(it))

image_opt = mi.render(scene, spp=512)

# Preview the final image
mi.util.convert_to_bitmap(image_opt)
plt.imshow(image_opt)
plt.show()

# # Get the meshtext bitmap
# bitmap = sceneparams['opt_texture']

# # Write the bitmap to a file
# bitmap.write('./', 'PNG')