# SUMMARY OF CONVERTING OF GETTING THE CAMERA MATRIX FROM COLMAP AND INTO THE SENSORS 
# The coordinates of the camera can be found in the image.txt file of the colmap output
# in the format of  ; IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
# QW, QX, QY, QZ is the quaternion and the TX, TY, TZ is the translation vector
# We want to take the QW, QX, QY, QZ and convert it into a 3x3 matrix
# get TX, TY, TZ and concatnate to the end giving us the camera matrix
# fill out the bottom 4 qith a 0001 and create a mistiba 4f vector and pass it into the sensor

# https://stackoverflow.com/questions/69210497/how-to-visualize-colmap-export-images-txt-in-blender


#WE NEED READ THE IMAGE.TXT FILE?
# TO DO: insert code that reads teh image txt file and gets the quarnion matrix values 
image_txt = "/Users/jadeybabey/Desktop/ERSP/south-building/sparse/images.txt" # insert path to images

#funtion that converts the quarternion to roation

import numpy as np

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


def main(): 
    with open(image_txt, "r") as f:
        lines = f.readlines()
        for index, temp_line in enumerate(lines):
            line = temp_line.strip()
            if line.endswith('JPG'):
                #print (line)
                temp = line.split(' ')

                qw = float(temp[1])
                qx = float(temp[2])
                qy = float(temp[3])
                qz = float(temp[4])
              
                tx = float(temp[5])
                ty = float(temp[6])
                tz = float(temp[7])

                rotation_matrix = quaternion_rotation_matrix ([qw,qx,qy,qz])
                print (rotation_matrix)
                # initalize translation matrix
                trans_matrix = np.array([[tx], [ty], [tz]])
                #print (trans_matrix)

                # take TX, TY, TZ and concatnate to the end giving us the camera matrix
                # along axis 1 
                matrix = np.concatenate((rotation_matrix,trans_matrix),axis = 1)
                #print (matrix)

                # fill out the bottom 4 qith a 0001 and create a mistiba 4f vector and pass it into the sensor
                bottom_4 =  np.array([[0, 0, 0, 1]])
                mistuba_vector = np.concatenate((matrix,bottom_4),axis = 0)

                #print(mistuba_vector)


# BASICALLY, THIS CODE LOOPS THORUGH THE THE IMAGE.TXT FILE AND GET THE CAMERA MATRIX FOR EACH IMAGE INPUTED 
# NOW WE WANT TO JUST INPUT IT INTO THE MITSUBA SENSOR 
if __name__ == '__main__':
    main()

            













