# ======================================================================================================================
# GLAcier Bed EsTimation by varying basal stress. [GLABET]
# ======================================================================================================================


# ======================================================================================================================

#                           Author:
#                        Saugat Paudel
#       M.S. by Research in Glaciology, Kathmandu University, Nepal.
#              Model created as a part of thesis requirement
#                   Email: saugat.email@gmail.com

# ======================================================================================================================


import time
import numpy as np
import rasterio
from shapely.ops import cascaded_union
from shapely.geometry import shape, mapping, LineString
from rasterio.mask import mask
import fiona
import subprocess
import os
import math
import csv


# ======================================================================================================================
__author__ = 'Saugat Paudel'
__version__ = '1.2'
__email__ = 'saugat.email@gmail.com'
__date__ = '6 March 2017'
# ======================================================================================================================


# ======================================================================================================================
# PARAMETERS TO CHANGE.
# ======================================================================================================================

# Paths

# Full Path to the glacier outline shapefile.
glacierOutlineFullPath = 'Place the path to the glacier outline shapefile here.'

# Full Path to the glacier DEM file. The DEM should preferably have same extent as the outline.
demFullPath = 'Place the path to the glacier DEM here.'

# Full path to the desired output folder.
outputFolder = 'Place the path to the output folder here.'

# Parameters
density = 900  # In kilograms per cubic meters. Default: 900
f = 0.8  # Dimensionless. Default: 0.8
g = 9.81  # Acceleration due to gravity in meters per seconds squared. Default: 9.81

# ======================================================================================================================
# ======================================================================================================================


# ======================================================================================================================
# Other variables. Not necessary to change.
# ======================================================================================================================

# Path to 'nnbathy' interpolation algorithm
'''
The nnbathy algorithm is included within the model folder. See glabet/nnbathy. In windows it may be necessary 
to change the slashes in the folder structure. Also it may be necessary to put '.exe' at the end. This depends on how 
the nnbathy algorithm was compiled.

Look in the 'nnbathy' help/readme to find out more about compilation
'''

# Give the relative path to the 'nnbathy' interpolation algorithm here. Edit 'nnbathy/nnbathy' part.
nnPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nnbathy/nnbathy')


# Output File Paths.
contourFullPath = os.path.join(outputFolder, 'contour.shp')
slopeFullPath = os.path.join(outputFolder, 'slope.tif')
sampledCSV = os.path.join(outputFolder, 'ungriddedCSV.txt')
finalCSV = os.path.join(outputFolder, 'finalCSV.txt')
thicknessRaster = os.path.join(outputFolder, 'Thickness.tif')
bedTopoFullPath = os.path.join(outputFolder, 'BedTopo.tif')
bufferedOutlineFullPath = os.path.join(outputFolder, 'bufferedOutline.shp')

# ======================================================================================================================
# FUNCTION DECLARATIONS. ALL REQUIRED FUNCTIONS ARE BELOW THIS LINE.
# ======================================================================================================================


# Check for the necessary files. This does not check the existence of the output folder.
def checkFiles():
    if os.path.exists(glacierOutlineFullPath) is False:
        print('Glacier outline not found!! Model will now exit.')
        exit()
    elif os.path.exists(demFullPath) is False:
        print('DEM not found!! Model will now exit.')
        exit()
    else:
        pass
    return None


# Check if the model was previously run and if there is existing data in the output folder.
def cleanUp():
    if os.path.exists(outputFolder):
        if os.path.dirname(glacierOutlineFullPath) == outputFolder or os.path.dirname(demFullPath) == outputFolder:
            print('Input data cannot be in the output folder. Please give a different folder path.')
            print('Model will now exit.')
            exit()
        print('\nThe output folder already exists.\n')
        print('WARNING: The output folder will be DELETED and RECREATED. \n'
              'Backup all the data contained in this folder.')
        decision = input('\nContinue (y/n) ? ')
        if decision == 'y':
            subprocess.call(['rm', '-r', outputFolder])
            os.mkdir(outputFolder)
        else:
            print('output folder not changed. Model will now exit.')
            exit()
    else:
        os.mkdir(outputFolder)
    return print('\nOutput folder is created .. OK!')


# Create contour from the given DEM. For this function to work 'gdal' libraries needs to be installed.
def createContour(pathToDEMFile, outputPath, contourInterval):
    subprocess.call(
        ['gdal_contour', '-a', 'elevation', '-i', str(contourInterval), '-f', 'ESRI Shapefile', pathToDEMFile,
         outputPath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return print('Contour shapefile is created ... OK!')


# Create slope from the given DEM. For this function to work 'gdal' libraries needs to be installed.
def createSlope(pathToDEMFile, outputPath):
    subprocess.call(['gdaldem', 'slope', pathToDEMFile, outputPath, '-of', 'GTiff', '-compute_edges'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return print('Slope raster is created ... OK!')


# Get the exterior coordinates of the glacier outline shapefile.
def getPolygonExteriorCoords(fionaFeatures):
    polygonExteriorCoords = []
    polygon = shape(fionaFeatures['geometry'])
    if polygon.geom_type == 'Polygon':
        polygonExteriorCoords.append(LineString(list(polygon.exterior.coords)))
    elif polygon.geom_type == 'MultiPolygon':
        for individualPolygons in polygon:
            polygonExteriorCoords.append(LineString(list(individualPolygons.exterior.coords)))
    else:
        print('WARNING: at function "getPolygonExteriorCoords", supported geometry not found!! Returning empty list.')
        pass
    return polygonExteriorCoords


# Get the interior coordinates of the glacier outline shapefile.
# This is needed in cases where the glacier has internal rings inside the boundary. Compare 'ponkar' and 'mera' glacier
# to see the differences
def getPolygonInteriorCoords(fionaFeature):
    interiorRings = []
    polygon = shape(fionaFeature['geometry'])
    if polygon.geom_type == 'Polygon':
        for interiorCoords in polygon.interiors:
            interiorRings.append(LineString(interiorCoords))
    elif polygon.geom_type == 'MultiPolygon':
        for individualPolygons in polygon:
            for interiorCoords in individualPolygons.interiors:
                interiorRings.append(LineString(interiorCoords))
    else:
        print('WARNING: at function "getPolygonInteriorCoords", supported geometry not found!! Returning empty list.')
        pass
    return interiorRings


# Read the metadata of the given DEM. Includes all necessary information such as resolution, extents, etc.
def getRasterMetadata(rasterFileFullPath):
    with rasterio.open(rasterFileFullPath) as rasterFile:
        rasterMetadata = rasterFile.meta.copy()
    return rasterMetadata


# Check if the raster file is in projected coordinate system. Model does not work for GCS projection.
# TODO: Check shapefile as well.
def checkRasterProjection(rasterFileFullPath):
    rasterMetaData = getRasterMetadata(rasterFileFullPath)
    rasterCRS = rasterMetaData['crs']
    print('DEM is not in projected Coordinate system. Please make sure it is! Model will now exit ...') or exit() \
        if rasterCRS.is_projected is False else None
    return print('Raster projected seem to be OK!')


# Get cosine values when input is in degrees.
def cosd(angleInDegrees):
    cosD = math.cos(math.radians(angleInDegrees))
    return cosD


# Get sine values when input is in degrees.
def sind(angleInDegrees):
    sinD = math.sin(math.radians(angleInDegrees))
    return sinD


# Get the mean raster slope. This is used in the estimation of the basal stress.
def getMeanRasterSlope(rasterioRasterObject, shapelyPolygonObject, returnMetaData=False):
    rasterMetaData = rasterioRasterObject.meta.copy()
    polygon = [mapping(shapelyPolygonObject)]
    maskedRaster, maskedRasterAffine = mask(rasterioRasterObject, polygon, nodata=np.nan)
    rasterMetaData.update({
        'driver': 'GTiff',
        'height': maskedRaster.shape[1],
        'width': maskedRaster.shape[2],
        'transform': maskedRasterAffine
    })
    meanRasterSlope = np.nanmean(maskedRaster)
    if returnMetaData:
        return meanRasterSlope, rasterMetaData
    else:
        return meanRasterSlope


# The changing outline of the glacier.
def createBufferedOutlines(inputShapefile, outputShapefile, rasterFileFullPath, bufferedist):
    with fiona.open(inputShapefile) as shapefile:
        driver = shapefile.driver
        crs = shapefile.crs
        schema = {'properties': {'ID': 'int', 'area': 'float', 'tau': 'float'}, 'geometry': 'MultiPolygon'}
        errorMsg = 'Given shapefile of the glacier outline contains more than 1 feature, ' \
                   'please ensure that the file has only one geometry \nModel will now exit...'
        print(errorMsg) or exit() if shapefile.__len__() != 1 else None
        with fiona.open(outputShapefile, 'w', crs=crs, schema=schema, driver=driver) as output:
            with rasterio.open(rasterFileFullPath) as raster:
                for features in shapefile:
                    outlineshape = shape(features['geometry'])
                    unionedpoly = cascaded_union(outlineshape)
                    initialArea = outlineshape.area
                    fid = 0
                    while cascaded_union(outlineshape.buffer(bufferedist)).type != 'GeometryCollection':
                        tau = (2.7 * 10 ** 4) * (((initialArea - unionedpoly.area) /
                                                  cosd(getMeanRasterSlope(raster, unionedpoly))) ** 0.106)
                        output.write({
                            'properties':
                                {
                                    'ID': fid,
                                    'area': initialArea - unionedpoly.area,
                                    'tau': tau,
                                },
                            'geometry': mapping(unionedpoly)
                        })
                        unionedpoly = (cascaded_union(outlineshape.buffer(bufferedist)))
                        outlineshape = outlineshape.buffer(bufferedist)
                        fid += 1
    return print('Buffered outline is created ... OK!')


# Pick raster values at given coordinates
def sampleRaster(rasterioRasterObject, samplingCoordX, samplingCoordY, band=1):
    rasterData = rasterioRasterObject.read(band)
    indexX, indexY = rasterioRasterObject.index(samplingCoordX, samplingCoordY)
    rasterValue = rasterData[indexX][indexY]
    return rasterValue


# Averaged slope at given coordinate. See description for more details.
def averagedSlopeSample(rasterObject, samplingCoordinateX, samplingCoordinateY, first=5, second=10, third=20, band=1):
    """
    Returns single averaged raster value at sampled point. Average is done at 3x3, 5x5 and 7x7 raster grid.

    The averaging is done based on the following:
        1. average a 7x7 grid if value is less than 'first'. Default: 5 (in degrees for slope)
        2. average a 5x5 grid if value is less than 'second'. Default: 10 (in degrees for slope)
        3. average a 3x3 grid if value is less than 'third'. Default: 20 (in degrees for slope)
        4. do not average if value exceeds 'third'. ie use the exact slope value
    The grid is defined in the 'grid3x3', 'grid5x5', and 'grid7x7' variables in the function itself. modifying these
    variables will give average at that grid.
    i.e grid3x3 = [2, 5] ==> will give values averaged at:
                             2 * 2 + 1 by 2 * 5 + 1 grid
                              i.e 5 x 11 grid.
    Defaults:
        grid3x3 = [1, 1] => implies 2 * 1 + 1,  2 * 1 + 1 = 3x3 grid.
        grid5x5 = [2, 2] => implies 2 * 2 + 1,  2 * 3 + 1 = 5x5 grid.
        grid7x7 = [3, 3] => implies 2 * 3 + 1,  2 * 3 + 1 = 7x7 grid.

    :param rasterObject: rasterio raster object. open the slope raster with rasterio.
    :param samplingCoordinateX: x-coordinate of the point from where the value of slope is required.
    :param samplingCoordinateY: y-coordinate of the point from where the value of slope is required.
    :param band: The band of raster. Default is 1.
    :param first: Value less than for which averaging is done with a 7x7 grid
    :param second: Value less than for which averaging is done with a 5x5 grid
    :param third:
    :returns: returns single averaged raster value at sampled point. Average is done at 3x3, 5x5 and 7x7 raster grid.
    """
    sumSlope = 0
    grid3x3 = [1, 1]
    grid5x5 = [2, 2]
    grid7x7 = [3, 3]
    rasterBand = rasterObject.read(band)
    indexX, indexY = rasterObject.index(samplingCoordinateX, samplingCoordinateY)
    slopeValue = rasterBand[indexX][indexY]
    if slopeValue <= first:
        count = 0
        for xIndices in range(indexX - (grid7x7[0]), indexX + (grid7x7[0] + 1)):
            for yIndices in range(indexY - (grid7x7[1]), indexY + (grid7x7[1] + 1)):
                if 0 <= xIndices <= rasterBand.shape[0] and 0 <= yIndices <= rasterBand.shape[1]:
                    try:
                        if rasterBand[xIndices][yIndices] == rasterObject.get_nodatavals() \
                                or rasterBand[xIndices][yIndices] == np.nan:
                            pass
                        else:
                            count += 1
                            sumSlope = sumSlope + np.nansum(rasterBand[xIndices][yIndices])
                    except IndexError:
                        print('WARNING: Index Exceeded at', samplingCoordinateX, samplingCoordinateY,
                              'resulting index was ', indexX, indexY)
                else:
                    pass
        try:
            averageSlope = sumSlope / count
        except ZeroDivisionError:
            print('WARNING: No Slope data found for ', grid7x7[0], 'x', grid7x7[1],
                  ' grid. Returning the exact slope at sampling points')
            averageSlope = slopeValue.astype(np.float64)
    elif slopeValue < second:
        count = 0
        for xIndices in range(indexX - (grid5x5[0]), indexX + (grid5x5[0] + 1)):
            for yIndices in range(indexY - (grid5x5[1]), indexY + (grid5x5[1] + 1)):
                if 0 <= xIndices <= rasterBand.shape[0] and 0 <= yIndices <= rasterBand.shape[1]:
                    try:
                        if rasterBand[xIndices][yIndices] == rasterObject.get_nodatavals() \
                                or rasterBand[xIndices][yIndices] == np.nan:
                            pass
                        else:
                            count += 1
                            sumSlope = sumSlope + np.nansum(rasterBand[xIndices][yIndices])
                    except IndexError:
                        print('WARNING: Index Exceeded at', samplingCoordinateX, samplingCoordinateY,
                              'resulting index was ', indexX, indexY)
                else:
                    pass
        try:
            averageSlope = sumSlope / count
        except ZeroDivisionError:
            print('WARNING: No Slope data found for ', grid5x5[0], 'x', grid5x5[1],
                  ' grid. Returning the exact slope at sampling points')
            averageSlope = slopeValue.astype(np.float64)
    elif slopeValue < third:
        count = 0
        for xIndices in range(indexX - (grid3x3[0]), indexX + (grid3x3[0] + 1)):
            for yIndices in range(indexY - (grid3x3[1]), indexY + (grid3x3[1] + 1)):
                if 0 <= xIndices < rasterBand.shape[0] and 0 <= yIndices < rasterBand.shape[1]:
                    try:
                        if rasterBand[xIndices][yIndices] == rasterObject.get_nodatavals() \
                                or rasterBand[xIndices][yIndices] == np.nan:
                            pass
                        else:
                            count += 1
                            sumSlope = sumSlope + np.nansum(rasterBand[xIndices][yIndices])
                    except IndexError:
                        print('WARNING: Index Exceeded at', samplingCoordinateX, samplingCoordinateY,
                              'resulting index was ', indexX, indexY)
                else:
                    pass
        try:
            averageSlope = sumSlope / count
        except ZeroDivisionError:
            print('WARNING: No Slope data found for ', grid3x3[0], 'x', grid3x3[1],
                  ' grid. Returning the exact slope at sampling points')
            averageSlope = slopeValue.astype(np.float64)
    else:
        averageSlope = slopeValue.astype(np.float64)
    return averageSlope


def getSamplePoints(shapelyGeometryWithLineStrings1, shapelyGeometryWithLineStrings2):
    x, y = [], []
    for features in shapelyGeometryWithLineStrings1:
        outlineShape = shape(features['geometry'])
        for individualContours in shapelyGeometryWithLineStrings2:
            contourShape = shape(individualContours['geometry'])
            intersectionX, intersectionY = contourShape.intersection(outlineShape)
            x.append(intersectionX) if intersectionX else None
            y.append(intersectionY) if intersectionY else None
    return x, y


# After sampling the thickness values at few points, the sampled points are interpolated to get a gridded data.
# 'nnbathy' algorithm is used here.
def interpolateUngriddedData(inputCSVFile, outputCSVFile, metaData):
    gridWidth = metaData['width']
    gridHeight = metaData['height']
    xMin = metaData['transform'][2]
    xMax = xMin + metaData['transform'][0] * gridWidth
    yMax = metaData['transform'][5]
    yMin = yMax + metaData['transform'][4] * gridHeight
    cmd = str(gridWidth) + 'x' + str(gridHeight)
    with open(outputCSVFile, 'w') as interpolatedCSV:
        subprocess.call([nnPATH, '-i', inputCSVFile, '-n', cmd, '-x', str(xMax), str(xMin), '-y', str(yMax),
                         str(yMin), '-W', '0'], stdout=interpolatedCSV)
    return print('Interpolation is completed ... OK!')


# Create a raster file from the gridded thickness data.
def createRaster(griddedData, outputRasterFullPath, metaData):
    data = np.loadtxt(griddedData)
    metaData.update({'dtype': 'float64', 'nodata': np.nan})
    pointThicknessValues = np.fliplr(np.reshape(data[:, 2], (metaData['height'], metaData['width'])))
    with rasterio.open(outputRasterFullPath, 'w', **metaData) as thicknessObject:
        thicknessObject.write(pointThicknessValues, 1)
    return print('Thickness raster is created ... OK!')


# Subtract the thickness raster from DEM to get Bed Topography.
def subtractRasters(firstRasterFullPath, secondRasterFullPath, outputRasterFullPath, firstBand=1, secondBand=1):
    with rasterio.open(firstRasterFullPath) as firstRaster:
        with rasterio.open(secondRasterFullPath) as secondRaster:
            rasterMetaData = firstRaster.meta.copy()
            rasterMetaData.update({'dtype': 'float64', 'nodata': np.nan})
            with rasterio.open(outputRasterFullPath, 'w', **rasterMetaData) as outputRaster:
                firstRasterData = firstRaster.read(firstBand)
                secondRasterData = secondRaster.read(secondBand)
                firstRasterData[firstRasterData == firstRaster.get_nodatavals()] = np.nan
                secondRasterData[secondRasterData == secondRaster.get_nodatavals()] = np.nan
                outputData = firstRasterData - secondRasterData
                outputRaster.write(outputData, 1)
    return print('Bed topograpgy is created ... OK!')


def main():

    # Check if input files are ok.
    checkFiles()

    # Create output folder.
    cleanUp()

    # To get the time taken by the model
    startTime = time.clock()

    # Get raster meta data
    rasterMetaData = getRasterMetadata(demFullPath)

    # Check if the given raster is projected.
    checkRasterProjection(demFullPath)

    # Create contours.
    createContour(demFullPath, contourFullPath, rasterMetaData['transform'][0])

    # Create slope.
    createSlope(demFullPath, slopeFullPath)

    # Create buffered outlines
    createBufferedOutlines(glacierOutlineFullPath, bufferedOutlineFullPath,
                           slopeFullPath, rasterMetaData['transform'][4])

    # Main section, where the magic happens
    with fiona.open(bufferedOutlineFullPath) as bufferedPolygons, \
        fiona.open(contourFullPath) as contours, \
        rasterio.open(slopeFullPath) as slopeRaster, \
        open(sampledCSV, 'w') as ungriddedCSV, \
        fiona.open(glacierOutlineFullPath) as glacierOutline:
            writeCSV = csv.writer(ungriddedCSV, delimiter=' ')

            # Set the ice thickness at glacier boundary to zero.
            for features in glacierOutline:
                featureGeom = shape(features['geometry'])
                exteriorCoords = featureGeom.exterior.coords
                for points in exteriorCoords:
                    writeCSV.writerow([points[0], points[1], 0])
                interiorGeom = featureGeom.interiors
                for intFeat in interiorGeom:
                    interiorCoords = intFeat.coords
                    for points in interiorCoords:
                        writeCSV.writerow([points[0], points[1], 0])

            # Get ice thickness at various sampling points.
            for individualFeature in bufferedPolygons:
                externalLines = getPolygonExteriorCoords(individualFeature)
                internalLines = getPolygonInteriorCoords(individualFeature)
                for individualContour in contours:
                    intersectionPoints = []
                    singleContour = shape(individualContour['geometry'])
                    for individualLines in externalLines:
                        intersectionPoints.append(singleContour.intersection(individualLines))
                    if not internalLines:
                        for individualLines in internalLines:
                            intersectionPoints.append(singleContour.intersection(individualLines))
                    for points in intersectionPoints:
                        if points.is_empty is False:
                            if points.geom_type == 'Point':
                                slope = averagedSlopeSample(slopeRaster, points.xy[0][0], points.xy[1][0])
                                h = individualFeature['properties']['tau']/(density * f * sind(slope) * g)
                                writeCSV.writerow([points.xy[0][0], points.xy[1][0], h])
                            elif points.geom_type == 'MultiPoint':
                                for individualPoints in points:
                                    slope = averagedSlopeSample(slopeRaster, individualPoints.xy[0][0],
                                                                individualPoints.xy[1][0])
                                    h = individualFeature['properties']['tau'] / (density * f * sind(slope) * g)
                                    writeCSV.writerow([individualPoints.xy[0][0], individualPoints.xy[1][0], h])
                            else:
                                print('WARNING: Neither "Point" or "MultiPoint". Doing Nothing!!')
                                pass
    print('Sampling is completed ... OK!')

    # Interpolate the un-gridded data
    interpolateUngriddedData(sampledCSV, finalCSV, rasterMetaData)

    # Create thickness raster.
    createRaster(finalCSV, thicknessRaster, rasterMetaData)

    # Create Bed Topography
    subtractRasters(demFullPath, thicknessRaster, bedTopoFullPath)

    # Total time taken by the model
    print('\nTotal time elapsed = ', time.clock() - startTime, ' seconds')

if __name__ == '__main__':
    main()
