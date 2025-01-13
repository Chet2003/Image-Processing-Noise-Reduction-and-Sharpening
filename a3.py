import sys
import numpy as np
import warp as wp
from PIL import Image

wp.init()
device = "cpu"
wp.config.cache_kernels = False

@wp.kernel
def produceBlurredL(
    warpInitialImage: wp.array(dtype=float, ndim=2),
    warpBlurredImage: wp.array(dtype=float, ndim=2),
    warpBlurredImageCount: wp.array(dtype=float, ndim=2),
    kernelSize: int,
    height: int, 
    width: int):

    i, j = wp.tid()

    kernelRange = int(kernelSize / 2)

    for yVar in range(-kernelRange, kernelRange + 1):
        for xVar in range(-kernelRange, kernelRange + 1):
            if (j + xVar) >= 0 and (i + yVar) >= 0 and (j + xVar) < width and (i + yVar) < height:
                warpBlurredImage[i, j] += (warpInitialImage[i + yVar, j + xVar])
                warpBlurredImageCount[i, j] += 1.0   
        
    if warpBlurredImageCount[i, j] > 0.0:
        warpBlurredImage[i, j] /= warpBlurredImageCount[i, j]
    else:
        warpBlurredImage[i, j] = 0.0

@wp.kernel
def produceBlurredRGB(
    warpInitialImage: wp.array(dtype=float, ndim=3),
    warpBlurredImage: wp.array(dtype=float, ndim=3),
    warpBlurredImageCount: wp.array(dtype=float, ndim=3),
    kernelSize: int,
    height: int, 
    width: int):

    i, j, k = wp.tid()

    kernelRange = int(kernelSize / 2)

    for yVar in range(-kernelRange, kernelRange + 1):
        for xVar in range(-kernelRange, kernelRange + 1):
            if (j + xVar) >= 0 and (i + yVar) >= 0 and (j + xVar) < width and (i + yVar) < height:
                warpBlurredImage[i, j, k] += (warpInitialImage[i + yVar, j + xVar, k])
                warpBlurredImageCount[i, j, k] += 1.0   
        
    if warpBlurredImageCount[i, j, k] > 0.0:
        warpBlurredImage[i, j, k] /= warpBlurredImageCount[i, j, k]
    else:
        warpBlurredImage[i, j, k] = 0.0

@wp.kernel
def produceEdgeL(
    warpInitialImage: wp.array(dtype=float, ndim=2),
    warpBlurredImage: wp.array(dtype=float, ndim=2),
    warpEdgeImage: wp.array(dtype=float, ndim=2)
):
    i,j = wp.tid()
    warpEdgeImage[i, j] = warpInitialImage[i, j] - warpBlurredImage[i, j]

@wp.kernel
def produceEdgeRGB(
    warpInitialImage: wp.array(dtype=float, ndim=3),
    warpBlurredImage: wp.array(dtype=float, ndim=3),
    warpEdgeImage: wp.array(dtype=float, ndim=3)
):
    i, j, k = wp.tid()
    warpEdgeImage[i, j, k] = warpInitialImage[i, j, k] - warpBlurredImage[i, j, k]

@wp.kernel
def unmaskedSharpeningL(
    warpInitialImage: wp.array(dtype=float, ndim=2),
    warpEdgeImage: wp.array(dtype=float, ndim=2),
    warpSharpenedImage: wp.array(dtype=float, ndim=2), 
    sigma: float
):

    i, j = wp.tid()
    warpSharpenedImage[i, j] = warpInitialImage[i, j] + (sigma * warpEdgeImage[i, j])

@wp.kernel
def unmaskedSharpeningRGB(
    warpInitialImage: wp.array(dtype=float, ndim=3),
    warpEdgeImage: wp.array(dtype=float, ndim=3),
    warpSharpenedImage: wp.array(dtype=float, ndim=3), 
    sigma: float
):

    i, j, k = wp.tid()
    warpSharpenedImage[i, j, k] = warpInitialImage[i, j, k] + (sigma * warpEdgeImage[i, j, k])

@wp.kernel
def medianFilterL(
    warpInitialImage: wp.array(dtype=float, ndim=2),
    warpMedianImage: wp.array(dtype=float, ndim=2),
    warpNeighbourCount: wp.array(dtype=int, ndim=2),    
    kernelSize: int,
    height: int, 
    width: int, 
):   

    i, j = wp.tid()
    
    neighbourhood = wp.vector(dtype=float, length=200)

    kernelRange = int(kernelSize / 2)

    for yVar in range(-kernelRange, kernelRange + 1):
        for xVar in range(-kernelRange, kernelRange + 1):
            if (j + xVar) >= 0 and (i + yVar) >= 0 and (j + xVar) < width and (i + yVar) < height:   
                neighbourhood[warpNeighbourCount[i,j]] = warpInitialImage[i + yVar, j + xVar]
                warpNeighbourCount[i,j] += 1

    for a in range(warpNeighbourCount[i,j]):
        for b in range(warpNeighbourCount[i,j]):
            if neighbourhood[a] > neighbourhood[b]:
                temp = neighbourhood[a]
                neighbourhood[a] = neighbourhood[b]
                neighbourhood[b] = temp
    
    warpMedianImage[i,j] = neighbourhood[int(warpNeighbourCount[i,j] / 2)]

@wp.kernel
def medianFilterRGB(
    warpInitialImage: wp.array(dtype=float, ndim=3),
    warpMedianImage: wp.array(dtype=float, ndim=3),
    warpNeighbourCount: wp.array(dtype=int, ndim=3),    
    kernelSize: int,
    height: int, 
    width: int, 
):   

    i, j, k = wp.tid()
    
    neighbourhood = wp.vector(dtype=float, length=200)

    kernelRange = int(kernelSize / 2)

    for yVar in range(-kernelRange, kernelRange + 1):
        for xVar in range(-kernelRange, kernelRange + 1):
            if (j + xVar) >= 0 and (i + yVar) >= 0 and (j + xVar) < width and (i + yVar) < height:   
                neighbourhood[warpNeighbourCount[i,j,k]] = warpInitialImage[i + yVar, j + xVar, k]
                warpNeighbourCount[i,j,k] += 1

    for a in range(warpNeighbourCount[i,j,k]):
        for b in range(warpNeighbourCount[i,j,k]):
            if neighbourhood[a] > neighbourhood[b]:
                temp = neighbourhood[a]
                neighbourhood[a] = neighbourhood[b]
                neighbourhood[b] = temp
    
    warpMedianImage[i,j,k] = neighbourhood[int(warpNeighbourCount[i,j,k] / 2)]

image = Image.open(sys.argv[4])

if image.mode == "RGBA":
    image = image.convert('RGB')
    
print(image.format)
print(image.size)
print(image.mode)

numpyInitialArr = np.asarray(image, dtype='float32')
print(numpyInitialArr.shape)

sigma = float(sys.argv[3])
kernelSize = int(sys.argv[2])
height = numpyInitialArr.shape[0]
width = numpyInitialArr.shape[1]

if sys.argv[1] == '-s':

    warpInitalArray = wp.from_numpy(numpyInitialArr, dtype=float, device=device)
    warpBlurredArray = wp.zeros(numpyInitialArr.shape, dtype=float, device=device)
    warpBlurredArrayCount = wp.zeros(numpyInitialArr.shape, dtype=float, device=device)
    warpEdgeArray = wp.zeros(numpyInitialArr.shape, dtype=float, device=device)
    warpSharpenedArray = wp.zeros(numpyInitialArr.shape, dtype=float, device=device)

    if image.mode == "L":
        wp.launch(kernel=produceBlurredL,
            dim = (height, width),
            inputs=[warpInitalArray, warpBlurredArray, warpBlurredArrayCount, kernelSize, height, width],
            device=device)        

        wp.launch(kernel=produceEdgeL,
            dim = (height, width),
            inputs=[warpInitalArray, warpBlurredArray, warpEdgeArray],
            device=device)

        wp.launch(kernel=unmaskedSharpeningL,
            dim = (height, width),
            inputs=[warpInitalArray, warpEdgeArray, warpSharpenedArray, sigma],
            device=device)

    if image.mode == "RGB":
        wp.launch(kernel=produceBlurredRGB,
            dim = (height, width, 3),
            inputs=[warpInitalArray, warpBlurredArray, warpBlurredArrayCount, kernelSize, height, width],
            device=device)        

        wp.launch(kernel=produceEdgeRGB,
            dim = (height, width, 3),
            inputs=[warpInitalArray, warpBlurredArray, warpEdgeArray],
            device=device)

        wp.launch(kernel=unmaskedSharpeningRGB,
            dim = (height, width, 3),
            inputs=[warpInitalArray, warpEdgeArray, warpSharpenedArray, sigma],
            device=device)

    numpySharpenedArray = np.asarray(wp.array.numpy(warpSharpenedArray), dtype='float32')         

    sharpened_clipped = np.clip(numpySharpenedArray, 0.0, 255.0)
    sharpened_image = Image.fromarray(np.uint8(sharpened_clipped))

    sharpened_image.save(sys.argv[5])

    print("Sharpened image saved as " + sys.argv[5])

if sys.argv[1] == '-n':

    if(kernelSize > 13):
        print("ERROR: The inputted kernel size is too big! Please use a smaller kernel size!")
        exit()

    warpInitalArray = wp.from_numpy(numpyInitialArr, dtype=float, device=device)
    warpMedianArray = wp.zeros(numpyInitialArr.shape, dtype=float, device=device)
    warpNeighbourCount = wp.zeros(numpyInitialArr.shape, dtype=int, device=device)

    if(image.mode == 'L'):
        wp.launch(kernel=medianFilterL,
            dim= (height, width),
            inputs=[warpInitalArray, warpMedianArray, warpNeighbourCount, kernelSize, height, width],
            device=device)   
    
    if(image.mode == 'RGB'):
        wp.launch(kernel=medianFilterRGB,
            dim= (height, width, 3),
            inputs=[warpInitalArray, warpMedianArray, warpNeighbourCount, kernelSize, height, width],
            device=device)   

    numpyMedianArray = np.asarray(wp.array.numpy(warpMedianArray), dtype='float32')

    median_image = Image.fromarray(np.uint8(numpyMedianArray))

    median_image.save(sys.argv[5])

    print("Median image saved as " + sys.argv[5])
