Edge Based Tracking library
===================================================

What is EBT?
--------------

EBT is a library of C++ classes that implement edge based object tracking in robotics and vision, using textureless object detection and tracking of the 3D pose.  

Detection and tracking schemes are coherently integrated in a particle filtering framework on the special Euclidean group, SE(3), in which the visual tracking problem is tackled by maintaining multiple hypotheses of the object pose. For textureless object detection, an efficient chamfer matching is employed so that a set of coarse pose hypotheses is estimated from the matching between 2D edge templates of an object and a query image. Particles are then initialized from the coarse pose hypotheses by randomly drawing based on costs of the matching. To ensure the initialized particles are at or close to the global optimum, an annealing process is performed after the initialization. While a standard edge-based tracking is employed after the annealed initialization, this library emploies a refinement process to establish improved correspondences between projected edge points from the object model and edge points from an input image.

how to generate template:

cd build
cmake ..
make
cp ../out/para_line_matcher.txt .
cp ../out/para_line_fitter.txt .
cp ../out/para_template_line_fitter.txt .

modify the Intrinsics_normal.xml with the camera parameter.
create a directory with object like [MBRFA30-2-P6], and we use this as the example.
put the obj file of the object in [MBRFA30-2-P6].
cp para_line_matcher.txt and para_line_fitter.txt to [MBRFA30-2-P6].

run
./ETGDemo -o MBRFA30-2-P6

then you can generate the template file.

