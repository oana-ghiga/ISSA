# # Exercise 10
# b_left, a_left = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
# b_right, a_right = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)
#
# left_top_y = int(0)
# left_top_x = int((left_top_y - b_left) / a_left)
# left_top = (int(left_top_x), int(left_top_y))
#
# left_bottom_y = int(height - 1)
# left_bottom_x = int((left_bottom_y - b_left) / a_left)
# left_bottom = (int(left_bottom_x), int(left_bottom_y))
#
# right_top_y = int(0)
# right_top_x = int((right_top_y - b_right) / a_right)
# right_top = (int(right_top_x), int(right_top_y))
#
# right_bottom_y = int(height - 1)
# right_bottom_x = int((right_top_y - b_right) / a_right)
# right_bottom = (int(right_bottom_x), int(right_bottom_y))
#
# # Not working for some reason
# if np.abs(left_top[0]) < 10 ** 8:
#     cv2.line(frame, left_top, left_bottom, (200, 0, 0), width)
#
# if np.abs(right_top[0]) < 10 ** 8:
#     cv2.line(frame, right_top, right_bottom, (100, 0, 0), width)
#
# if ret is False:
#     break
#
# # Show final frame if 'f' key is pressed
# key = cv2.waitKey(1)
# if key == ord('f'):
#     # a. Create a blank frame
#     blank_frame = np.zeros_like(frame)
#
#     # b. Draw only the left line onto the blank frame
#     cv2.line(blank_frame, (left_top[0], left_top[1]), (left_bottom[0], left_bottom[1]), (255, 0, 0), 3)
#
#     # c. Map the blank frame to the trapezoid
#     magic_matrix_left_final = cv2.getPerspectiveTransform(screen_points, np.float32(points_trapezoid))
#
#     # d. Warp the blank frame
#     left_line_trapezoid_final = cv2.warpPerspective(blank_frame, magic_matrix_left_final, (width, height))
#
#     # e. Get coordinates of the white pixels for the left line
#     left_line_coords_final = np.argwhere(left_line_trapezoid_final[:, :, 0] > 0)
#
#     # f. Repeat for the right line
#     blank_frame_right = np.zeros_like(frame)
#     cv2.line(blank_frame_right, (right_top[0], right_top[1]), (right_bottom[0], right_bottom[1]), (0, 255, 0),
#              3)
#     magic_matrix_right_final = cv2.getPerspectiveTransform(screen_points, np.float32(points_trapezoid))
#     right_line_trapezoid_final = cv2.warpPerspective(blank_frame_right, magic_matrix_right_final,
#                                                      (width, height))
#     right_line_coords_final = np.argwhere(right_line_trapezoid_final[:, :, 1] > 0)
#
#     # g. Color the original frame
#     colored_frame = np.zeros_like(frame)
#     colored_frame[left_line_coords_final[:, 0], left_line_coords_final[:, 1]] = [50, 50, 250]  # Red
#     colored_frame[right_line_coords_final[:, 0], right_line_coords_final[:, 1]] = [50, 250, 50]  # Green
#
# # Exercise 12
# it runs in real time