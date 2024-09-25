# from datetime import date

# print(str(date.today()))

# import cv2

# # Đường dẫn tới video gốc
# input_video_path = 'trash/videos/video_split/video1_00006.avi'

# # Đường dẫn đến video mới sau khi lưu
# output_video_path = 'video1_00006.mp4'

# # Tạo đối tượng VideoCapture để đọc video gốc
# video_capture = cv2.VideoCapture(input_video_path)

# # Lấy các thông số kỹ thuật của video gốc
# fps = video_capture.get(cv2.CAP_PROP_FPS)
# frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Tạo đối tượng VideoWriter để ghi video mới
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video mới là MP4
# video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# # Đọc từng khung hình của video gốc, xử lý (nếu cần), và ghi vào video mới
# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break  # Hết video, thoát khỏi vòng lặp

#     # Xử lý khung hình ở đây nếu muốn

#     # Ghi khung hình vào video mới
#     video_writer.write(frame)

# # Giải phóng các đối tượng
# video_capture.release()
# video_writer.release()
# cv2.destroyAllWindows()

from logs import setup_logger

logg = setup_logger('test', debug=True)
logg.debug('debug')