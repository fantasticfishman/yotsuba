https://github.com/fantasticfishman/yotsuba 

Yotsuba is our attempt at creating a beginner-friendly inverse rendering pipeline, built off the Mitsuba 3(get it) rendering library.

In other words, our goal is to intake input images, and 1) get a first guess of what the object might look like, 2)optimize the geometry of that object to make it easier to work with, 3) optimize the texture of that object to make the colors the same as in the pictures. 

We have 3 main branches to try and accomplish these, because the issue is that the initial guess software we are using(COLMAP), doesn’t really allow us to have a good guess to start the pipeline, so that needs to be fixed/better software has to be found before we merge everything together into a full pipeline. To fix that, we idealized and assumed we got the proper output from COLMAP in steps 2 and 3. Step 1 is in Jade’s branch, 2 is in Kevin’s branch, and 3 is in Justin’s branch.

Before we get to that, we introduce the tools we are using. The scripts we are using are written in python, and the renderer we are using is called Mitsuba 3. https://mitsuba.readthedocs.io/en/stable/
Read up on this first. Like actually. 

https://colmap.github.io/index.html 
This is the initial guess software we are using.

Other requirements:
Pip3 install mitsuba
Pip3 install matplotlib
Brew install llvm

The general idea of the pipeline goes: 

1- input images into colmap
2- get initial guess from colmap, and camera angles
3- use gradient descent optimization to optimize a blank sphere to the initial guess mesh, to make it easier to work with(using largesteps code, done by Kevin in his branch)
4-UV map the mesh so we can put a texture on it
5-Put a blank texture on the mesh, take the camera angles from COLMAP, and then compare the input images and the mesh from identical camera angles, and then optimize the mesh to have the proper colors.

1-2 are covered in Jade’s branch
3 is covered in Kevin’s branch
4-5 is covered in Justin’s branch

Try looking in the branches and running some of the scripts in the main directory to see how things work. 

https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html
This tutorial is a small scale demo of what we are trying to do, try to get it running so you can get an idea of how things work. 

