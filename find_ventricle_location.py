# -*- coding: utf-8 -*-
'''
Code to find center of left ventricle using only DICOM data
Part of 8th place solution. Author: ZFTurbo
Initial code based on http://www.dclunie.com/dicom3tools/workinprogress/dcpost.cc

Will generate:
<output_data_path>/geometry.json - extracted geometry data from DICOM files
<output_data_path>/center_points.json - coordinates of left ventricle in JSON format
<output_data_path>/center_find/*.jpg - debug JPG files
'''

import numpy as np
import os
import cv2
import re
import json
import glob
import dicom
import simplejson


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom)*db + b1


def getPositionOrientationSpacingAndSizeFromGeom(geom):
    row_dircos_x = geom['ImageOrientationPatient'][0]
    row_dircos_y = geom['ImageOrientationPatient'][1]
    row_dircos_z = geom['ImageOrientationPatient'][2]

    col_dircos_x = geom['ImageOrientationPatient'][3]
    col_dircos_y = geom['ImageOrientationPatient'][4]
    col_dircos_z = geom['ImageOrientationPatient'][5]

    nrm_dircos_x = row_dircos_y * col_dircos_z - row_dircos_z * col_dircos_y
    nrm_dircos_y = row_dircos_z * col_dircos_x - row_dircos_x * col_dircos_z
    nrm_dircos_z = row_dircos_x * col_dircos_y - row_dircos_y * col_dircos_x

    pos_x = geom['ImagePositionPatient'][0]
    pos_y = geom['ImagePositionPatient'][1]
    pos_z = geom['ImagePositionPatient'][2]

    rows = geom['Rows']
    cols = geom['Columns']

    row_spacing = geom['PixelSpacing'][0]
    col_spacing = geom['PixelSpacing'][1]

    row_length = rows*row_spacing
    col_length = cols*col_spacing

    return row_dircos_x, row_dircos_y, row_dircos_z, col_dircos_x, col_dircos_y, col_dircos_z, \
        nrm_dircos_x, nrm_dircos_y, nrm_dircos_z, pos_x, pos_y, pos_z, rows, cols, \
        row_spacing, col_spacing, row_length, col_length


def rotate(dst_row_dircos_x, dst_row_dircos_y, dst_row_dircos_z,
	dst_col_dircos_x, dst_col_dircos_y, dst_col_dircos_z,
	dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z,
	src_pos_x, src_pos_y, src_pos_z):
    dst_pos_x = dst_row_dircos_x * src_pos_x + dst_row_dircos_y * src_pos_y + dst_row_dircos_z * src_pos_z
    dst_pos_y = dst_col_dircos_x * src_pos_x + dst_col_dircos_y * src_pos_y + dst_col_dircos_z * src_pos_z
    dst_pos_z = dst_nrm_dircos_x * src_pos_x + dst_nrm_dircos_y * src_pos_y + dst_nrm_dircos_z * src_pos_z
    return dst_pos_x, dst_pos_y, dst_pos_z


def line_plane_intersection(point_plane_x, point_plane_y, point_plane_z,
                            point1_line_x, point1_line_y, point1_line_z,
                            point2_line_x, point2_line_y, point2_line_z,
                            plane_nrm_x, plane_nrm_y, plane_nrm_z):
    part_1_x = (point_plane_x - point1_line_x)
    part_1_y = (point_plane_y - point1_line_y)
    part_1_z = (point_plane_z - point1_line_z)
    part_2 = np.dot([part_1_x, part_1_y, part_1_z], [plane_nrm_x, plane_nrm_y, plane_nrm_z])
    line_dir_x = point2_line_x - point1_line_x
    line_dir_y = point2_line_y - point1_line_y
    line_dir_z = point2_line_z - point1_line_z

    part_3 = np.dot([line_dir_x, line_dir_y, line_dir_z], [plane_nrm_x, plane_nrm_y, plane_nrm_z])
    # print(part_2, part_3)
    d_koeff = part_2/part_3
    cross_x = d_koeff*line_dir_x + point1_line_x
    cross_y = d_koeff*line_dir_y + point1_line_y
    cross_z = d_koeff*line_dir_z + point1_line_z
    # print(cross_x, cross_y, cross_z)
    return cross_x, cross_y, cross_z


def get_line_intersection(gdst, gsrc):
    dst_row_dircos_x, dst_row_dircos_y, dst_row_dircos_z, dst_col_dircos_x, dst_col_dircos_y, dst_col_dircos_z, \
        dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z, dst_pos_x, dst_pos_y, dst_pos_z, dst_rows, dst_cols, \
        dst_row_spacing, dst_col_spacing, dst_row_length, dst_col_length \
        = getPositionOrientationSpacingAndSizeFromGeom(gdst)
    src_row_dircos_x, src_row_dircos_y, src_row_dircos_z, src_col_dircos_x, src_col_dircos_y, src_col_dircos_z, \
        src_nrm_dircos_x, src_nrm_dircos_y, src_nrm_dircos_z, src_pos_x, src_pos_y, src_pos_z, src_rows, src_cols, \
        src_row_spacing, src_col_spacing, src_row_length, src_col_length \
        = getPositionOrientationSpacingAndSizeFromGeom(gsrc)

    pos_x = [0, 0, 0, 0, 0, 0, 0, 0]
    pos_y = [0, 0, 0, 0, 0, 0, 0, 0]
    pos_z = [0, 0, 0, 0, 0, 0, 0, 0]

    # TLHC is what is in ImagePositionPatient

    pos_x[0] = src_pos_x
    pos_y[0] = src_pos_y
    pos_z[0] = src_pos_z

    # TRHC

    pos_x[1] = src_pos_x + src_row_dircos_x*src_row_length
    pos_y[1] = src_pos_y + src_row_dircos_y*src_row_length
    pos_z[1] = src_pos_z + src_row_dircos_z*src_row_length

    # BRHC

    pos_x[2] = src_pos_x + src_row_dircos_x*src_row_length + src_col_dircos_x*src_col_length
    pos_y[2] = src_pos_y + src_row_dircos_y*src_row_length + src_col_dircos_y*src_col_length
    pos_z[2] = src_pos_z + src_row_dircos_z*src_row_length + src_col_dircos_z*src_col_length

    # BLHC

    pos_x[3] = src_pos_x + src_col_dircos_x*src_col_length
    pos_y[3] = src_pos_y + src_col_dircos_y*src_col_length
    pos_z[3] = src_pos_z + src_col_dircos_z*src_col_length

    for i in range(4):
        # Line intersection with plane
        pos_x[4+i], pos_y[4+i], pos_z[4+i] = line_plane_intersection(dst_pos_x, dst_pos_y, dst_pos_z,
                                                           pos_x[i], pos_y[i], pos_z[i],
                                                           pos_x[(i+1)%4], pos_y[(i+1)%4], pos_z[(i+1)%4],
                                                           dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z)
        # print(pos_x[4+i], pos_y[4+i], pos_z[4+i])


    # First 4 - points projection
    # Last 4 - intersection
    row_pixel = [0, 0, 0, 0, 0, 0, 0, 0]
    col_pixel = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        '''
        // we want to view the source slice from the "point of view" of
        // the target localizer, i.e. a parallel projection of the source
        // onto the target
        // do this by imaging that the target localizer is a view port
        // into a relocated and rotated co-ordinate space, where the
        // viewport has a row vector of +X, col vector of +Y and normal +Z,
        // then the X and Y values of the projected target correspond to
        // row and col offsets in mm from the TLHC of the localizer image
        // move everything to origin of target
        '''

        pos_x[i] -= dst_pos_x
        pos_y[i] -= dst_pos_y
        pos_z[i] -= dst_pos_z

        # The rotation is easy ... just rotate by the row, col and normal
        # vectors ...

        pos_x[i], pos_y[i], pos_z[i] = rotate(
            dst_row_dircos_x, dst_row_dircos_y, dst_row_dircos_z,
            dst_col_dircos_x, dst_col_dircos_y, dst_col_dircos_z,
            dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z,
            pos_x[i], pos_y[i], pos_z[i])

        # DICOM coordinates are center of pixel 1\1

        col_pixel[i] = int(pos_x[i]/dst_col_spacing + 0.5)
        row_pixel[i] = int(pos_y[i]/dst_row_spacing + 0.5)

    # Most distant points
    xx = 4
    yy = 5
    max_dist = 0
    for i in range(4, 8):
        for j in range(i+1, 8):
            dist = (row_pixel[i] - row_pixel[j])*(row_pixel[i] - row_pixel[j]) +\
                   (col_pixel[i] - col_pixel[j])*(col_pixel[i] - col_pixel[j])
            if dist > max_dist:
                max_dist = dist
                xx = i
                yy = j

    # Return 2 most distance points of intersection plane
    return row_pixel[xx], col_pixel[xx], row_pixel[yy], col_pixel[yy]


def find_intersections_point(gsax, g2ch, g4ch):
    point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col = get_line_intersection(gsax, g2ch)
    point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col = get_line_intersection(gsax, g4ch)

    intersect = seg_intersect(np.array([point_ch2_1_row, point_ch2_1_col]),
                              np.array([point_ch2_2_row, point_ch2_2_col]),
                              np.array([point_ch4_1_row, point_ch4_1_col]),
                              np.array([point_ch4_2_row, point_ch4_2_col]))

    return intersect.tolist(), \
           point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col, \
           point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_grayscale_with_increase_brightness_fast(im, incr):
   min = np.amin(im.astype(float))
   max = np.amax(im.astype(float))
   out = incr*((im - min) * (255)) / (max - min)
   out[out > 255] = 255
   out = out.astype(np.uint8)

   return out

def calculate_consistent_square(img, geom, center):
    square_dim = 200.0
    row_spacing = geom['PixelSpacing'][0]
    col_spacing = geom['PixelSpacing'][1]

    row_px = int(round(square_dim / row_spacing))
    col_px = int(round(square_dim / col_spacing))

    top_left_corner = [round(center[0] - (square_dim / 2.0)), round(center[1] - (square_dim / 2.0))]
    row_side_pixels = round(square_dim / row_px)
    col_side_pixels = round(square_dim / col_px)

    img = img[top_left_corner:top_left_corner+row_side_pixels, top_left_corner:top_left_corner+col_side_pixels]
    return img




def draw_center_for_check(dcm_path, id, sax, point, points, geom):
    debug_folder = '/scratch/gaas0012/calc/center_find'#os.path.join('..', 'calc', 'center_find')
    if not os.path.isdir(debug_folder):
        os.mkdir(debug_folder)
    ds = dicom.read_file(dcm_path)
    img = convert_to_grayscale_with_increase_brightness_fast(ds.pixel_array, 1)
    img = calculate_consistent_square(img, geom, (int(round(point[1], 0)), int(round(point[0], 0))))
    #cv2.circle(img, (int(round(point[1], 0)), int(round(point[0], 0))), 5, 255, 3)

    #cv2.line(img, (points[1], points[0]), (points[3], points[2]), 127, thickness=2)
    #cv2.line(img, (points[5], points[4]), (points[7], points[6]), 127, thickness=2)
    # show_image(img)
    cv2.imwrite(os.path.join(debug_folder, str(id) + '_' + sax + '.jpg'), img)


def get_centers_for_test(id, geom, debug):
    # print(geom)
    center = dict()
    ch2_el = ''
    ch4_el = ''
    for el in geom:
        matches = re.findall("(2ch_\d+)", el)
        if len(matches) > 0:
            ch2_el = el
        matches = re.findall("(4ch_\d+)", el)
        if len(matches) > 0:
            ch4_el = el

    if ch2_el != '' and ch4_el != '':
        for el in geom:
            if el != ch2_el and el != ch4_el:
                print('Start extraction for test {} sax {}'.format(id, el))
                try:
                    center[el], point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col, \
                    point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col \
                        = find_intersections_point(geom[el], geom[ch2_el], geom[ch4_el])

                    if debug == 1:
                        draw_center_for_check(geom[el]['Path'], id, el, center[el],
                                      (point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col,
                                       point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col), geom)
                except:
                    print('Problem with calculation here!')
                    center[el] = [-1, -1]

    else:
        print('Test {} miss 2ch or 4ch view of heart!'.format(id))

    return center

# Read file with DCM geometry
def read_geometry_file():
    json_path = '/scratch/gaas0012/calc/geometry.json'#os.path.join('..', 'calc', 'geometry.json')
    geom = dict()
    if os.path.isfile(json_path):
        f = open(json_path, 'r')
        geom = simplejson.load(f)
        f.close()
    keys = list(geom.keys())
    for el in keys:
        geom[int(el)] = geom[el]
    for el in keys:
        geom.pop(el, None)
    return geom


def get_all_centers(start, end, debug):
    centers = dict()
    geom = read_geometry_file()
    #print(geom)
    for i in range(start, end+1):
        centers[i] = get_centers_for_test(i, geom[i], debug)
    return centers


def store_centers(centers, path):
    f = open(path, 'w')
    simplejson.dump(centers, f)
    f.close()


def get_start_end_patients(type, input_data_path):
    split = -1
    if type == 'all':
        path = os.path.join(input_data_path, 'train')
        dirs = os.listdir(path)
        max = int(dirs[0])
        for d in dirs:
            if int(d) > max:
                max = int(d)
        split = max
        path = os.path.join(input_data_path, 'validate')
        dirs += os.listdir(path)
    else:
        path = os.path.join(input_data_path, type)
        print(path)
        dirs = os.listdir(path)
    min = int(dirs[0])
    max = int(dirs[0])
    for d in dirs:
        if int(d) < min:
            min = int(d)
        if int(d) > max:
            max = int(d)
    return min, max, split


def find_geometry_params(start, end, split, input_data_path, output_data_path):
    if not os.path.isdir(output_data_path):
        os.mkdir(output_data_path)
    json_path = os.path.join(output_data_path, 'geometry.json')
    store = dict()
    for i in range(start, end+1):
        store[i] = dict()
        type = 'train'
        path = os.path.join(input_data_path, type, str(i), 'study', '*')
        dcm_files = glob.glob(path)
#        print('Total files found for test ' + str(i) + ': ' + str(len(dcm_files)))

        for d_dir in dcm_files:
            print('Read single DCMs for test' + str(i) + ': ' + d_dir)
            dfiles = os.listdir(d_dir)
            for dcm in dfiles:
                sax_name = os.path.basename(d_dir)
                dcm_path = os.path.join(d_dir, dcm)
                if (os.path.isfile(dcm_path)):
                    print('Reading file: ' + dcm_path)
                    ds = dicom.read_file(dcm_path)
                    store[i][sax_name] = dict()
                    store[i][sax_name]['ImageOrientationPatient'] = (ds.ImageOrientationPatient[0],
                                                                     ds.ImageOrientationPatient[1],
                                                                     ds.ImageOrientationPatient[2],
                                                                     ds.ImageOrientationPatient[3],
                                                                     ds.ImageOrientationPatient[4],
                                                                     ds.ImageOrientationPatient[5])
                    store[i][sax_name]['ImagePositionPatient'] = (ds.ImagePositionPatient[0],
                                                                     ds.ImagePositionPatient[1],
                                                                     ds.ImagePositionPatient[2])
                    store[i][sax_name]['PixelSpacing'] = (ds.PixelSpacing[0],
                                                          ds.PixelSpacing[1])
                    store[i][sax_name]['SliceLocation'] = (ds.SliceLocation)
                    store[i][sax_name]['SliceThickness'] = (ds.SliceThickness)
                    store[i][sax_name]['Rows'] = (ds.Rows)
                    store[i][sax_name]['Columns'] = (ds.Columns)
                    store[i][sax_name]['Path'] = dcm_path
                    break

    f = open(json_path, 'w')
    simplejson.dump(store,f)
    f.close()

# Put train and validate folders here
input_data_path = '../data'
# Results will be stored in this folder
output_data_path = '/scratch/gaas0012/calc'

start, end, split = get_start_end_patients('train', input_data_path)
find_geometry_params(start, end, split, input_data_path, output_data_path)
centers = get_all_centers(start, end, 1)
out_path = os.path.join(output_data_path, 'center_points.json')
store_centers(centers, out_path)
