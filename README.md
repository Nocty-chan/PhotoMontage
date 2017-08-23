# Photomontage Project #

This is an implementation of the Interactive Digital Photomontage system presented in [Agarwala, Aseem, et al. "Interactive digital photomontage." ACM Transactions on Graphics (TOG). Vol. 23. No. 3. ACM, 2004](http://grail.cs.washington.edu/projects/photomontage/photomontage.pdf).

This version implements only the `DESIGNATED_SOURCE` data penalty and `COLORS`,`GRADIENTS` and `COLORS_AND_GRADIENTS` interaction penalties. This allows the user to composite images by painting the zone he wants to keep. The images are patched together using graph-cut optimization.

This project was the final paired-project for the Computer Vision class (INF552) at Ecole polytechnique (X) taught by Pr. Renaud Keriven. The webpage for the class can be accessed [here](http://www.enseignement.polytechnique.fr/informatique/INF552/projets.htm).

## How to use ##

* Clone the repository
* `cmake CMakeLists.txt && make`
* `./PROJET`

* An openCV window continaing the first image appears, using the mouse and the left button, you can paint on the zone you want to keep from the first image. Right-click on the image when you're done in order to go to the following image.
* Strokes have different colors depending on the image they were defined on, in order for the user to remember the previous zones he chose.
* Right-clicking on the last image launches the graph-cut optimization.
* When graph-cut optimization is done, the result image appears along with the pixel map that shows the origin of each pixel in the result composite.
* Tap any key to exit the program.

## Example results ##

### Family photo ###

Initial photos:


![Photo 1](https://github.com/Nocty-chan/PhotoMontage/blob/master/images/famille1.jpg?raw=true)
![Photo 2](https://github.com/Nocty-chan/PhotoMontage/blob/master/images/famille2.jpg?raw=true)
![Photo 3](https://github.com/Nocty-chan/PhotoMontage/blob/master/images/famille3.jpg?raw=true)
![Photo 4](https://github.com/Nocty-chan/PhotoMontage/blob/master/images/famille1.jpg?raw=true)

Result photos:


![Result](https://github.com/Nocty-chan/PhotoMontage/blob/master/results/composite-result-family.png?raw=true)
![Origin of pixels](https://github.com/Nocty-chan/PhotoMontage/blob/master/results/composite-origin-family.png?raw=true)
