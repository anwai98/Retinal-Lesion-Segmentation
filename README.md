# Retinal-Lesion-Segmentation
Retinal Lesions (Microaneurysms, Hard Exudates, Soft Exudates, Hemorrhages) Segmentation using Deep Learning Pipeline and Image Processing &amp; Machine Learning Pipeline

Dataset used - IDRiD Challenge Dataset

How to play with the codes?
1. Image Processing - To extract the reasonable candidates with a high sensitivity. (My insight - The implemented method isn't the finest. Simple Image Smoothing with some Contrast Enhancement, Study of Different Color Space and Morphological Operations should do the job with ease. Vessel Segmentation plays an important role too, besides Opitc Disc Segmentation, only if it works 100% perfect).
2. Machine Learning - To extract features from the candidates available to obtain a multi-class classification. (My insight - The implemented method is perfect, keeping in mind that the candidate feature extraction has to be commendable. Try different Image Processing and Image Feature Extraction weapons by your own to experiment with the pipeline)
3. Deep Learning - To segment the lesions into a binary class segmentation outcome. (My insight - Understand the loss functions and the different architectures).

Note : Comments are provided for explanation (almost). For more information, contact me!
