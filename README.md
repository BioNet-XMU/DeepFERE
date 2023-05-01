# DeepFURE
DeepFURE is a deep learning platform for high-resolution reconstruction of mass spectrometry imaging incorporated with mutimodal fusion.
Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of DeepFURE

<div align=center>
<img src="https://user-images.githubusercontent.com/70273368/230378358-129af693-2e52-4197-a037-600dce0b6bac.png" width="800" height="550" /><br/>
</div>

__Schematic overflow of the DeepFURE model__. The registration module registers H&E image to MSI to obtain registered H&E image. 
The mapping module generates a HR MSI. The raw MSI data is embedded to low-dimension MSI, which is bicubically interpolated to form a HR MSI. 
Two HR MSIs are concatenated and inputted into the fusion module to generate the fused HR image. 
Three loss functions including mapping loss, correlation loss and reconstruction loss are used to optimize the DeepFURE to generate the HR MSI.

# Requirement

    python == 3.5, 3.6 or 3.7
    
    pytorch == 1.8.2
    
    opencv == 4.5.3
    
    matplotlib == 2.2.2

    numpy >= 1.8.0
    
    umap == 0.5.1
    
# Quickly start

## Input

DeepFURE models provide MSI high-resolution reconstructions in two scenarios: embedding data and single-ion image.

(1) High-resolution reconstruction for embedding data: (a) LR embedding data; (b) HR H&E image; (c) n_factor.

(2) High-resolution reconstruction for embedding data: (a) LR embedding data; (b) LR single ion image; (c) HR H&E image; (d) n_factor.

Here, 3-dimension MSI data with two-dimensional shape [X*Y,3], single ion image with two-dimensional shape [X,Y], where X and Y represent the pixel numbers of horizontal and vertical coordinates of MSI data; H&E image with three-dimensional shape [H,W,3], where H and W represent the pixel numbers of horizontal and vertical coordinates of H&E image;  n_factor: the user-definded mangification. 

## Run DeepFURE model

cd to the DeepFURE fold

If you want to perfrom DeepFURE for *16 high-resolution restruction of embeddin data, taking fetus mouse brain section as an example, run:
    
    python run.py --mode embedding --input_MSIEfile .../example/mouse_brain_umap.txt --input_HEfile .../example/HE.png --input_shape 40 36 --n_factor 16 
    
If you want to perfrom DeepFURE for *16 high-resolution restruction of single ion image, taking fetus mouse brain section as an example, run:

    python run.py --mode ion --input_MSIEfile .../example/mouse_brain_umap.txt --input_MSIIfile .../example/mouse_brain_ion505.txt --input_HEfile .../example/HE.png --input_shape 40 36 --n_factor 16 

## Contact

Please contact me if you have any help: gl5121405@gmail.com



