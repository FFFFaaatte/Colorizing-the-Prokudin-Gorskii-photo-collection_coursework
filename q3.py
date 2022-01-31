import cv2
import numpy as np

img = cv2.imread('master-pnp-prok-00400-00458a.tif',cv2.IMREAD_UNCHANGED) # read  in unchanged mode (it's not colorful by default)

# master-pnp-prok-00400-00458a.tif train [42, 5, 87, 31]
# master-pnp-prok-01800-01833a.tif mosque [55, -3, 124, -1]
# master-pnp-prok-01800-01886a.tif amir [50, 24, 104, 42]

img= img/65535 # scale the value of pixels between 0 and 1. 65535 is the maximum number which can be shown using 16 bits
length= np.shape(img)[0]//3 # we are going to divide the image into 3 equal parts so we calculate number of rows then divide it by 3
# we cut the image vertically, all of parts has a same size
b= img[0:length-1,]
g= img[length:2*length-1,]
r=img[2*length:3*length-1,]

# merging three channels, this will make a bad quality image
imgres= cv2.merge((b,g,r))

b2=b # b2, g2 and r2 are copies of channels. (we are going to change r,g,b)
g2=g
r2=r
# as margins of photo don't contain exact data we crop it.(note that we just don't consider
# this margins in calculating movements but we will add this parts at last)
# as the size of three images doesnt differ very largely I've considered this values as constants. (otw they can depend on size of given images)
imgres= imgres[256:imgres.shape[0]-256,256:imgres.shape[1]-256]

b,g,r= cv2.split(imgres) # splitting the image to its channels

# this is actually an image pyramid
# we resize each layer to half in each step
layer1_2_b= cv2.resize(b, (0, 0), fx=0.5, fy=0.5)
layer1_4_b= cv2.resize(layer1_2_b, (0, 0), fx=0.5, fy=0.5)
layer1_8_b= cv2.resize(layer1_4_b, (0, 0), fx=0.5, fy=0.5)
layer1_16_b= cv2.resize(layer1_8_b, (0, 0), fx=0.5, fy=0.5)


layer1_2_g= cv2.resize(g, (0, 0), fx=0.5, fy=0.5)
layer1_4_g= cv2.resize(layer1_2_g, (0, 0), fx=0.5, fy=0.5)
layer1_8_g= cv2.resize(layer1_4_g, (0, 0), fx=0.5, fy=0.5)
layer1_16_g= cv2.resize(layer1_8_g, (0, 0), fx=0.5, fy=0.5)


layer1_2_r= cv2.resize(r, (0, 0), fx=0.5, fy=0.5)
layer1_4_r= cv2.resize(layer1_2_r, (0, 0), fx=0.5, fy=0.5)
layer1_8_r= cv2.resize(layer1_4_r, (0, 0), fx=0.5, fy=0.5)
layer1_16_r= cv2.resize(layer1_8_r, (0, 0), fx=0.5, fy=0.5)

# consider a channel as base and don not move it. I considered channel b is the base. then we
# take a background for 2 other channels and check different places to put g and r channel for the best match.
# result of each size (1/16, 1/8,..) will multiply by 2. Then we search a few number of pixels in the next(larger) size for the best match.
# we do this for all sizes until we reach to original size

def do_for_each_level(layer_b,layer_g,layer_r,g_x,g_y,r_x,r_y,margin,interval):
    movement=[] # array to store movements for any axis
    base= np.zeros([np.shape(layer_b)[0] + 2*margin, np.shape(layer_b)[1] + 2* margin]) # background for base layer
    base[margin:layer_b.shape[0] + margin, margin:layer_b.shape[1] + margin] = layer_b # put b layer at the center of the base
    background_g= np.zeros([np.shape(layer_g)[0] + 2*margin, np.shape(layer_g)[1] + 2*margin])   # background for g layer
    inf= float('inf') # infinity
    min_diff_g= inf # minimum of L1. this is assumed to be larger than any other value at first
    # looping to find best place for g layer

    for i in range (g_x-interval+margin,g_x+interval+margin+1): # looping for x index i --> x
      for j in range(g_y-interval+margin,g_y+interval+margin+1): # looping for y index j --> y
          background_g[i:layer_g.shape[0] + i, j:layer_g.shape[1] + j] = layer_g # put g layer first index in (i,j) index of  g background
          L1_g= np.abs(base - background_g) # calculate L1 parameter
          # just intersection is important
          L1_g= L1_g[margin:layer_b.shape[0] + margin, margin:layer_b.shape[1] + margin]
          sum1= np.sum(L1_g) # summation of the intersection part
          if min_diff_g>sum1: # if this new summation is less than last founded minimum then
            min_diff_g=sum1                       # replace it
            move_x_g= i   # set movement in x axis equal to i
            move_y_g = j  # set movement in y axis equal to j
    # we consider the movements based on the base layer and we ignore the added margins
    movement.append(move_x_g-margin)
    movement.append(move_y_g-margin)
    # doing the same thinf for channel r
    background_r= np.zeros([np.shape(layer_r)[0] + 2* margin, np.shape(layer_r)[1] + 2* margin])

    min_diff_r= inf
    for i in range (r_x-interval+margin,r_x+interval+margin+1):
        for j in range(r_y-interval+margin,r_y+interval+margin+1):
                background_r[i:layer_r.shape[0] + i, j:layer_r.shape[1] + j] = layer_r
                L1_r= np.abs(base - background_r)
                # just intersection is important
                L1_r= L1_r[margin:layer_b.shape[0] + margin, margin:layer_b.shape[1] + margin]
                sum2= np.sum(L1_r)
                if min_diff_r>sum2:
                    min_diff_r=sum2
                    move_x_r= i
                    move_y_r = j

    movement.append(move_x_r-margin)
    movement.append(move_y_r-margin)

    return movement # returns an array index 0 -> horizontal movement of g
    # 1 -> vertical movement of g
    # 2 -> horizontal movement of r
    # 3 -> vertical movement of r

def do_movements_on_img(movement_arr,margin,b,g,r): # function to apply calculatad movements on channels and combine
       # them to make a colorful imagr
    # add margins to movements again (we will crop this margins later)
    move_x_g= movement_arr[0]+margin
    move_y_g= movement_arr[1]+margin
    move_x_r= movement_arr[2]+margin
    move_y_r= movement_arr[3]+margin
    s0,s1= b.shape
    base = np.zeros([s0+ 2 * margin, s1 + 2 * margin])  # background for base layer
    base[margin:s0 + margin,margin:s1+ margin] = b  # put b layer at the center of the base

    background_g=np.zeros((s0 + 2*margin, s1+ 2*margin)) # background for g layer

    background_g[move_x_g:s0 + move_x_g, move_y_g:s1 + move_y_g] = g # apply movements on g layer

    background_r=np.zeros([s0 + 2*margin, s1 + 2*margin]) # background for r layer

    background_r[move_x_r:s0 + move_x_r, move_y_r:s1 + move_y_r] = r # apply movements on r layer

    img_res= cv2.merge((base, background_g, background_r)) # combine 3 channels to make a colorful image
    return img_res

movement_1_16= do_for_each_level(layer_b=layer1_16_b,layer_g=layer1_16_g,layer_r=layer1_16_r,g_x=0,g_y=0,r_x= 0,r_y=0,margin=10,interval=8)

# color_x , color_y means from where of original img we must start
# interval shows the domain we are going to search for the best match
for i in range(0, len(movement_1_16)):
    movement_1_16[i] =movement_1_16[i]*2
movement_1_8= do_for_each_level(layer_b=layer1_8_b,layer_g=layer1_8_g,layer_r=layer1_8_r,g_x=movement_1_16[0],g_y=movement_1_16[1],
                                r_x= movement_1_16[2],r_y=movement_1_16[3],margin=30,interval=1)
for i in range(0, len(movement_1_8)):
    movement_1_8[i] =movement_1_8[i]*2
movement_1_4= do_for_each_level(layer_b=layer1_4_b,layer_g=layer1_4_g,layer_r=layer1_4_r,g_x=movement_1_8[0],g_y=movement_1_8[1],
                                r_x= movement_1_8[2],r_y=movement_1_8[3],margin=50,interval=2)
for i in range(0, len(movement_1_4)):
    movement_1_4[i] =movement_1_4[i]*2
movement_1_2 =do_for_each_level(layer_b=layer1_2_b,layer_g=layer1_2_g,layer_r=layer1_2_r,g_x=movement_1_4[0],g_y=movement_1_4[1],
                                r_x= movement_1_4[2],r_y=movement_1_4[3],margin=100,interval=2)
for i in range(0, len(movement_1_2)):
    movement_1_2[i] =movement_1_2[i]*2
movement_1 =do_for_each_level(layer_b=b,layer_g=g,layer_r=r,g_x=movement_1_2[0],g_y=movement_1_2[1],
                                r_x= movement_1_2[2],r_y=movement_1_2[3],margin=180,interval=2)
# print(movement_1)
imgres= do_movements_on_img(movement_arr=movement_1,margin=180,b=b2,g=g2,r=r2)
# cropping th added margin
imgres= imgres[180:imgres.shape[0]-180,180:imgres.shape[1]-180]

# now we want to recognize margins of strange colors and remove them
# following variables will contain size of margins
row1=0 # up
row2=0 # down
col1=0 # left
col2=0 # right

# for loop ( a number to show what pixels we should check for margins)
tresh= int(imgres.shape[0]/10) # considered that images are length and width are not verryyy different

tr=400* (imgres.shape[0])/3000 #  for standard deviation . this is found by experiment

# consider sum of standard deviation of r,g,b lines and rows. we loop in an interval and find where this parameter is large
for i in range(1,tresh):
    s0,s1= b2.shape
    row1_b = (b2[i,])
    row2_b = (b2[s0 - i, :])
    col1_b = (b2[:, i])
    col2_b = (b2[:, s1 - i])
    row1_r = (r2[i,])
    row2_r = (r2[s0 - i, :])
    col1_r = (r2[:, i])
    col2_r = (r2[:, s1 - i])
    row1_g = (g2[i,])
    row2_g = (g2[s0 - i, :])
    col1_g = (g2[:, i])
    col2_g = (g2[:, s1 - i])

    std1 = np.sum(np.std((row1_g, row1_r, row1_b), axis=0))
    std2 = np.sum(np.std((row2_g, row2_r, row2_b), axis=0))
    std3 = np.sum(np.std((col1_g, col1_r, col1_b), axis=0))
    std4 = np.sum(np.std((col2_g, col2_r, col2_b), axis=0))

# in this way we will find largest probable indices:
    if std1>tr:
        row1=i
    if std2>tr:
        row2=i
    if std3>tr:
        col1=i
    if std4>tr:
        col2= i

# crop margins from indices we've just found
imgres= imgres[row1:s0-row2,col1:s1-col2]
imgres= (imgres*256).astype('uint8') # convert the image to 8 bit

# save the image as jpg
cv2.imwrite('res05-Train.jpg', imgres)