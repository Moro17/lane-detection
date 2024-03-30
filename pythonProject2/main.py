import cv2
import numpy as np

def convert_to_gray(color_pixel):
    # Convert a 3-element color pixel to a single grayscale value using the Luma formula
    gray_value = 0.299 * color_pixel[2] + 0.587 * color_pixel[1] + 0.114 * color_pixel[0]
    return int(gray_value)

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

while True:
    ret, frame = cam.read()

    # ret (bool): Return code of the 'read' operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)
    # frame (array): the actual frame as an array.
    #                Height x Width x3 ( 3 color, BGR ) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc;
    #
    if ret is False:
        break
# 1
    cv2.imshow('Original', frame)

# 2 shrink the frame

    scale_percent = 0.35  # percent of original size
    width_r = int(frame.shape[1] * scale_percent)
    height_r = int(frame.shape[0] * scale_percent)
    dim = (width_r, height_r)

    resized = cv2.resize(frame, dim)
    cv2.imshow('Small', resized)

# 3 conv to grayscale
    #2nd method to create grayscale image
    gray_frame = np.zeros((height_r, width_r), dtype=np.uint8)
 #   for y in range(height):
 #       for x in range(width):
 #           color_pixel = resized[y, x]
 #           gray_value = convert_to_gray(color_pixel)
 #           gray_frame[y, x] = gray_value

    resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', resized_gray)

# 4 we shrink the image to have only the road(we done need the background)

    trapezoid = np.zeros((height_r,width_r), dtype=np.uint8)
    upper_left = (int(width_r * 0.48), int(height_r * 0.75))
    upper_right = (int(width_r * 0.54), int(height_r * 0.75))
    lower_left = (int(width_r * 0), int(height_r * 1))
    lower_right = (int(width_r * 1), int(height_r * 1))
    white_trap = np.array([upper_left, upper_right, lower_right, lower_left], dtype=np.int32)
    trapezoid = cv2.fillConvexPoly(trapezoid, white_trap, 1)
    cv2.imshow('Trapezoid', trapezoid * 255)
    road = trapezoid * resized_gray
    cv2.imshow('Road', road)

# 5 stretching the image

    f_ul = (0, 0)
    f_ur = (width_r, 0)
    f_ll = (0, height_r)
    f_lr = (width_r, height_r)
    white_trap = np.float32(white_trap)
    n_frame = np.array([f_ul,f_ur,f_lr,f_ll], dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(white_trap, n_frame)
    top_down = cv2.warpPerspective(road, perspective_matrix, dim)
    cv2.imshow('Top-Down', top_down)

# 6 add blur

    n = 5
    blur = cv2.blur(top_down, ksize=(n, n))
    cv2.imshow('Blur', blur)

# 7 edge detection

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [+1, +2, +1]])

    sobel_horizontal = np.transpose(sobel_vertical)  # transpose

    blur = np.float32(blur)
    sobel_filter_1 = cv2.filter2D(blur, -1, sobel_vertical)
    sobel_filter_2 = cv2.filter2D(blur, -1, sobel_horizontal)
# Displaying images
    convert1 = cv2.convertScaleAbs(sobel_filter_1)
    convert2 = cv2.convertScaleAbs(sobel_filter_2)
    cv2.imshow('No.1', convert1)
    cv2.imshow('No.2', convert1)
    sobel_final = np.sqrt(sobel_filter_1**2 + sobel_filter_2**2)
    convert_final = cv2.convertScaleAbs(sobel_final)
    cv2.imshow('Sobel', convert_final)
# 8 binarize the frame
    threshold = int(200 / 2)
    ret, bi = cv2.threshold(convert_final,threshold,225, cv2.THRESH_BINARY)
    # maxval sets the maxvalue the pixel will get (black) and last paramter sets a functin 0/max
    # ret returns the optimal threshold value
    cv2.imshow('Binarized', bi)

# 9 Get the coordinates of street markings on each side of the road
    copy = bi.copy()
    a = int(width_r * 0.05)
    b = int(width_r * 0.95)
    bi[:, :a] = 0
    bi[:, b:] = 0

    half_1 = bi[:, 0:width_r // 2]
    half_2 = bi[:, (width_r // 2 + 1):]

    left_h = np.argwhere(half_1 > 0)
    right_h = np.argwhere(half_2 >0)

    left_xs = left_h[:, 1]
    left_ys = left_h[:, 0]
    right_xs = right_h[:, 1] + width_r//2  #Offset
    right_ys = right_h[:, 0]
    right_xs1 = right_h[:, 1]

# 10 10.	Find the lines that detect the edges of the lane

    b_left, a_left = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    b_right, a_right = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    left_top_y = 0
    left_bottom_y = height_r
    right_top_y = 0
    right_bottom_y = height_r

    left_top_x = (-b_left) / a_left
    left_bottom_x = (height_r - b_left) / a_left
    right_top_x = (- b_right) / a_right
    right_bottom_x = (height_r - b_right) / a_right

    # Bad Values
    if not (-1e8 <= left_top_x <= 1e8):
        left_top_x = 0
        left_top_y = 0
    if not (-1e8 <= left_bottom_x <= 1e8):
        left_bottom_x = 0
        left_bottom_y = height_r
    if not (-1e8 <= right_top_x <= 1e8):
        right_top_x = 0
        right_top_y = 0
    if not (-1e8 <= right_bottom_x <= 1e8):
        right_bottom_x = 0
        right_bottom_y = height_r

    left_top = (int(left_top_x), int(left_top_y))
    left_bottom = (int(left_bottom_x), int(left_bottom_y))
    right_top = (int(right_top_x), int(right_top_y))
    right_bottom = (int(right_bottom_x), int(right_bottom_y))

    cv2.line(bi, left_top, left_bottom, (200, 0, 0), 3)
    cv2.line(bi, right_top, right_bottom, (100, 0, 0), 3)
    cv2.line(bi, (width_r // 2, 0), (width_r // 2, height_r), (255, 0, 0), 1)
    cv2.imshow('Lines', bi)

# 11 add a red line which follow the dot line and a green line which follow the full line

    left_lines_frame = np.zeros(dim, dtype=np.uint8)
    right_lines_frame = np.zeros_like(resized)


    cv2.line(left_lines_frame, left_top, left_bottom, (255, 0, 0), 3)
    cv2.line(right_lines_frame, right_top,right_bottom, (0, 255, 0), 3)

    final_matrix = cv2.getPerspectiveTransform(n_frame, white_trap)

    top_down_left = cv2.warpPerspective(left_lines_frame, final_matrix, dim)
    top_down_right = cv2.warpPerspective(right_lines_frame, final_matrix, dim)

    copy_left = top_down_left.copy()
    copy_right = top_down_right.copy()
    copy_original = frame.copy()

    white_left = np.argwhere(top_down_left == 255)
    white_right = np.argwhere(top_down_right == 255)

    for coord in white_left:
        resized[coord[0], coord[1]] = [50, 50, 250]
    for coord in white_right:
        resized[coord[0], coord[1]] = [50, 250, 50]

    cv2.imshow('Final', resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()