import numpy
cimport numpy
cimport cython

ctypedef numpy.uint8_t DTYPE_t
ctypedef numpy.npy_int32 INT32_t 

@cython.boundscheck(False)
@cython.wraparound(False)

def map_face(
    numpy.ndarray[DTYPE_t, ndim=3] camera,
    numpy.ndarray[DTYPE_t, ndim=3] face,
    numpy.ndarray[INT32_t, ndim=1] bbox_frame,
    numpy.ndarray[INT32_t, ndim=1] bbox_face
    ):

    cdef int ix, iy, ix_frame, iy_frame, ix_face, iy_face
    cdef float px, py

    for iy in range(bbox_frame[3]):
        
        py = float(iy) / float(bbox_frame[3])
        iy_face = int(bbox_face[2] + py * bbox_face[3])
        iy_frame = bbox_frame[2] + iy
        iy_frame = max(0, min(iy_frame, 479))
        #iy_face = max(0, min(iy_face, 479))

        for ix in range(bbox_frame[1]):
            px = float(ix) / float(bbox_frame[1])                      
            ix_face = int(bbox_face[0] + px * bbox_face[1])       
            ix_frame = bbox_frame[0] + ix          
            ix_frame = max(0, min(ix_frame, 639)) 
            #ix_face = max(0, min(ix_face, 639))

            camera[iy_frame, ix_frame, :] = face[iy_face, ix_face, :]

    return camera

def map_face_check(
    numpy.ndarray[DTYPE_t, ndim=3] camera,
    numpy.ndarray[DTYPE_t, ndim=3] face,
    numpy.ndarray[INT32_t, ndim=1] bbox_frame,
    numpy.ndarray[INT32_t, ndim=1] bbox_face,
    numpy.ndarray[INT32_t, ndim=2] points
    ):

    cdef int ix, iy, ix_frame, iy_frame, ix_face, iy_face
    cdef float px, py

    cdef numpy.ndarray v0
    cdef numpy.ndarray v1
    cdef numpy.ndarray v2
    cdef numpy.ndarray P = numpy.zeros(2)

    cdef float dot00, dot01, dot02, dot11, dot12, invDenom, div_

    # A = points[0,:]
    # B = points[1,:]
    # C = points[2,:]

    v0 = points[2,:] - points[0,:]
    v1 = points[1,:] - points[0,:]

    dot00 = numpy.dot(v0, v0)
    dot01 = numpy.dot(v0, v1)
    dot11 = numpy.dot(v1, v1)

    div_ = (dot00 * dot11 - dot01 * dot01)
    if div_ == 0:
        div_ = 0.0001
    invDenom = 1 / div_

    for iy in range(bbox_frame[3]):
        
        py = float(iy) / float(bbox_frame[3])
        iy_face = int(bbox_face[2] + py * bbox_face[3])
        iy_frame = bbox_frame[2] + iy
        iy_frame = max(0, min(iy_frame, 479))
        #iy_face = max(0, min(iy_face, 479))

        for ix in range(bbox_frame[1]):
            px = float(ix) / float(bbox_frame[1])                      
            ix_face = int(bbox_face[0] + px * bbox_face[1])       
            ix_frame = bbox_frame[0] + ix          
            ix_frame = max(0, min(ix_frame, 639)) 
            #ix_face = max(0, min(ix_face, 639))

            # TRIANGLE CHECK
            P[0] = iy_frame
            P[1] = ix_frame
            v2 = P - points[0,:]
            dot02 = numpy.dot(v0, v2)
            dot12 = numpy.dot(v1, v2)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom

            if (u >= 0) and (v >= 0) and (u + v < 1):
                camera[iy_frame, ix_frame, :] = face[iy_face, ix_face, :]

    return camera

def map_avg(
    numpy.ndarray[DTYPE_t, ndim=3] camera,
    numpy.ndarray[DTYPE_t, ndim=1] avg,
    numpy.ndarray[INT32_t, ndim=1] bbox_frame,
    numpy.ndarray[INT32_t, ndim=1] bbox_face
    ):

    cdef int ix, iy, ix_frame, iy_frame

    for iy in range(bbox_frame[3]):
        
        iy_frame = bbox_frame[2] + iy
        iy_frame = max(0, min(iy_frame, 479))

        for ix in range(bbox_frame[1]):                         
            ix_frame = bbox_frame[0] + ix          
            ix_frame = max(0, min(ix_frame, 639)) 

            camera[iy_frame, ix_frame, :] = avg

    return camera

def map_avg_check(
    numpy.ndarray[DTYPE_t, ndim=3] camera,
    numpy.ndarray[DTYPE_t, ndim=1] avg,
    numpy.ndarray[INT32_t, ndim=1] bbox_frame,
    numpy.ndarray[INT32_t, ndim=1] bbox_face,
    numpy.ndarray[INT32_t, ndim=2] points
    ):

    cdef int ix, iy, ix_frame, iy_frame

    cdef numpy.ndarray v0
    cdef numpy.ndarray v1
    cdef numpy.ndarray v2
    cdef numpy.ndarray P = numpy.zeros(2)

    cdef float dot00, dot01, dot02, dot11, dot12, invDenom, div_

    v0 = points[2,:] - points[0,:]
    v1 = points[1,:] - points[0,:]

    dot00 = numpy.dot(v0, v0)
    dot01 = numpy.dot(v0, v1)
    dot11 = numpy.dot(v1, v1)

    div_ = (dot00 * dot11 - dot01 * dot01)
    if div_ == 0:
        div_ = 0.0001
    invDenom = 1 / div_

    for iy in range(bbox_frame[3]):
        
        iy_frame = bbox_frame[2] + iy
        iy_frame = max(0, min(iy_frame, 479))

        for ix in range(bbox_frame[1]):                         
            ix_frame = bbox_frame[0] + ix          
            ix_frame = max(0, min(ix_frame, 639)) 

            # TRIANGLE CHECK
            P[0] = iy_frame
            P[1] = ix_frame
            v2 = P - points[0,:]
            dot02 = numpy.dot(v0, v2)
            dot12 = numpy.dot(v1, v2)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom

            if (u >= 0) and (v >= 0) and (u + v < 1):
                camera[iy_frame, ix_frame, :] = avg
            
    return camera
