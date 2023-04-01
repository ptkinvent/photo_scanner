# Photo Scanning Tool #
In 2022, my family hired someone to scan about 3000 photos from physical photo albums in India. Unfortunately, instead
of using a scanner, this person photographed each photo using a Nikon DSLR against an orange background. Fortunately,
they did photograph them individually, with good lighting, and without glares. The purpose of this project is to isolate
the photographs from their backgrounds and output the highest-possible resolution photos for storing in our digital
family albums.

This project uses computer vision techniques to detect the 4 corners of the photograph within the image, then performs a
perspective shift to isolate the photograph.

This project was done in Python 3.8. The plotting script was helpful in discovering how to remove the orange pixels in
the images.
