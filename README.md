# DeepSURE
DeepSURE is a deep learning platform for high-resolution reconstruction of mass spectrometry imaging incorporated with mutimodal fusion.
Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of DeepSURE

<div align=center>
<img src="https://user-images.githubusercontent.com/70273368/230378358-129af693-2e52-4197-a037-600dce0b6bac.png" width="800" height="550" /><br/>
</div>

__Schematic overflow of the DeepSURE model__. The registration module registers H&E image to MSI to obtain registered H&E image. 
The mapping module generates a HR MSI. The raw MSI data is embedded to low-dimension MSI, which is bicubically interpolated to form a HR MSI. 
Two HR MSIs are concatenated and inputted into the fusion module to generate the fused HR image. 
Three loss functions including mapping loss, correlation loss and reconstruction loss are used to optimize the DeepSURE to generate the HR MSI.

# Requirement

    python == 3.5, 3.6 or 3.7
    
    pytorch == 1.8.2
    
    opencv == 4.5.3
    
    matplotlib == 2.2.2

    numpy >= 1.8.0
    
    umap == 0.5.1
    
# Quickly start

## Input

The input is the preprocessed MSI data with two-dimensional shape [X*Y,P], where X and Y represent the pixel numbers of horizontal and vertical coordinates of MSI data, and P represents the number of ions.
