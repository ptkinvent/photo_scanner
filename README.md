# Photo Cropping Tool #
In 2022, my family hired someone to scan about 3000 photos from previous family albums in India. Unfortunately, instead
of using a scanner, this person photographed each photo using a Nikon DSLR. Fortunately, they did remove each photo from
the albums and photograph them individually, with good lighting, and without glares. However, these photos were placed
on an ugly orange chair. The purpose of this project is to remove this ugly background from our family archives.

We do this using OpenCV to detect the 4 corners of the photograph within the image, preserving only the pixels within
those 4 corners, then performing a perspective shift to remove any rotation/tilt of the photograph.

This project was made using Python 3.8. It consists of two scripts -- one to plot the pixels of the original image in
RGB and HSV, and one to actually remove the background. The plotting script was helpful in discovering how to remove
the ugly orange chair from the images.
