# Instruction on how we create PixelShift Dataset
This readme file is only for readers who are interested in how our dataset is generated. 

# Step 1: read Sony ARW (Pixel Shift: 4 Raws for each scene), save the RGGB file as .mat 
    
    python generate_pixelshift.py 

# Step 2: Crop effective areas out of Pixelshift RGGB images
    
    matlab cropPixelShift.m
