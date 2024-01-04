# dataset :
## với label thì để từ 10-50 cũng đc nhưng ở code thì khi lấy ảnh 10 sẽ lấy tầm 8(16) ảnh ơt ngay trước để làm 1 video với frame từ 2-10 , nếu < 1 thì sẽ duplicate frame 1 như vậy để có 1 data đẹp ta nên tạo data như nào.

## Đầu tiên hướng dẫn cách cắt frame từ video trước đã
### từ video cắt thành các frame và detecr person
```sh
python extract_frame/video2frame.py
```
### có 1 số biến truyền vào như folder video hoặc chỉ chạy 1 video, nơi lưu video với frame đã vẽ lên số frame, nơi lưu frame + label.txt và cả ảnh debug là ảnh vẽ lên đó số frame, box để tiện cho việc chia lại các frame đó đúng label

## sau khi cắt frame, hãy xem video or folder debug ảnh để xem các frame nào thuộc class nào. cách lưu file thì readme có minh họa rồi, qua đó xem. Nhắc lại là image-rbg thừ 2 class để nguyên cho lành, chỉ chọn file txt thôi
Lưu ý: khi chia các frame từ 1 video thành 2 đoạn của 2 class thì nên xóa 1 số frame giữa vì ở trên có nói lí do rồi và ảnh thì giữa nguyên, không cần xóa
Khi extract video to frame, chọn label khác 2 label thì file label.txt sẽ là 0 x y h z 
cần chuyển thành theo đúng thì chạy code :
```sh
python trash/data_raw.py
```

Data được lưu tại folder trash. Trong đó có code để tạo 2 file testlist.txt và trainlist.txt
```sh
python trash/build_data_list.py 
```

# nếu bạn có video dài muốn cắt nhỏ hay convert video thì xem file : extract_frame/split_video.py . Code dễ hiểu nên nói tóm gọn là có 2 chức năng: tách 1 video thành nhiều video và convert video từ mp4 -> gì đó

# sau khi có 2 file txt train test list thì bắt tay vào train và test thôi
```sh
trong ./script có mấy file bash, truyền tham số thì tương tự mẫu là được 
```

