# Instruction on how we create PixelShift Dataset
# Step 1: read Sony ARW (Pixel Shift: 4 Raws for each scene), save the RGGB file as .mat 
    
    python generate_pixelshift.py 

# Step 2: Crop effective areas out of Pixelshift RGGB images
    
    matlab cropPixelShift.m
